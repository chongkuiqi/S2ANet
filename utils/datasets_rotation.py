# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""

import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from zipfile import ZipFile

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

from utils.augmentations import augment_hsv, letterbox, mixup,random_perspective_rotation
from utils.general import (LOGGER, NUM_THREADS,
                           x1y1x2y2x3y3x4y4n2x1y1x2y2x3y3x4y4,
                           xywhtheta2xywhtheta_n,
                           poly_to_rotated_box_np,
                           )
from utils.torch_utils import torch_distributed_zero_first

# Parameters
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = ['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp']  # include image suffixes
VID_FORMATS = ['asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'wmv']  # include video suffixes

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except Exception:
        pass

    return s


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {2: Image.FLIP_LEFT_RIGHT,
                  3: Image.ROTATE_180,
                  4: Image.FLIP_TOP_BOTTOM,
                  5: Image.TRANSPOSE,
                  6: Image.ROTATE_270,
                  7: Image.TRANSVERSE,
                  8: Image.ROTATE_90,
                  }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


def create_dataloader(path, imgsz, batch_size, stride, single_cls=False, hyp=None, augment=False, cache=False, pad=0.0,
                      rect=False, rank=-1, workers=8, quad=False, prefix='', shuffle=False):
    if rect and shuffle:
        LOGGER.warning('WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,  # augmentation
                                      hyp=hyp,  # hyperparameters
                                      rect=rect,  # rectangular batches
                                      cache_images=cache,
                                      single_cls=single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    # 由于image_weights默认都是False，因此loader一直都是InfiniteDataLoader
    # loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    # loader = InfiniteDataLoader  # only DataLoader allows for attribute updates
    loader = DataLoader  # only DataLoader allows for attribute updates
    # 在分布式训练模式下，sampler是distributed.DistributedSampler，并且shuffle=True，因此确实是打乱顺序的，
    #   但是传入loader的shuffle参数是False，并且由于采用了InfiniteDataLoader，实际上是没有打乱的
    # 在普通训练模式下，sampler是None，并且shuffle=True，因此传入loader的shuffle参数是True，
    #   如果loader是DataLoader，则会打乱顺序，但由于是InfiniteDataLoader，因此不会打乱顺序
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=True,
                  collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn), dataset


class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap, s

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


class LoadImagesAndLabels(Dataset):
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version

    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=''):
        
        self.img_size = img_size
        
        # 一般是True
        self.augment = augment
        self.hyp = hyp

        self.rect =  rect
        # mosaic增强只在打开数据增强的情况下进行
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                
                # p:'/home/lab/ckq/DOTA_split/train.txt'
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            
            self.img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}')

        # Check cache
        self.label_files = img2label_paths(self.img_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        
        # 
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache['version'] == self.cache_version  # same version
            assert cache['hash'] == get_hash(self.label_files + self.img_files)  # same hash
        except Exception:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache


        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupt"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {HELP_URL}'

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels, shapes = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)
        

        for i, label in enumerate(self.labels):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]

            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0


        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs, self.img_npy = [None] * n, [None] * n
        if cache_images:
            if cache_images == 'disk':
                self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + '_npy')
                self.img_npy = [self.im_cache_dir / Path(f).with_suffix('.npy').name for f in self.img_files]
                self.im_cache_dir.mkdir(parents=True, exist_ok=True)
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(NUM_THREADS).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                if cache_images == 'disk':
                    if not self.img_npy[i].exists():
                        np.save(self.img_npy[i].as_posix(), x[0])
                    gb += self.img_npy[i].stat().st_size
                else:
                    self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB {cache_images})'
            pbar.close()

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(verify_image_label, zip(self.img_files, self.label_files, repeat(prefix))),
                        desc=desc, total=len(self.img_files))
            for im_file, lb, shape, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [lb, shape]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupt"

        pbar.close()
        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, len(self.img_files)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            LOGGER.warning(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # not writeable
        return x

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *load_mosaic(self, random.randint(0, self.n - 1)))

        else:
            # Load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            shape = self.img_size  # final letterboxed shape
            # padding到方形尺寸
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                # labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
                labels[:, 1:] = x1y1x2y2x3y3x4y4n2x1y1x2y2x3y3x4y4(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
                
            if self.augment:
                img, labels = random_perspective_rotation(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        
        if self.augment:

            nl = len(labels)  

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    # labels[:, 2] = 1 - labels[:, 2]
                    labels[:, 2::2] = img.shape[0] - labels[:, 2::2]

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    # labels[:, 1] = 1 - labels[:, 1]
                    labels[:, 1::2] = img.shape[1] - labels[:, 1::2]

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout
        # print(img.dtype)
        
        # print(labels)
        nl = len(labels)  # number of labels
        # if nl:
        #     img1 = plot_rotate_boxes(img.copy(), labels)
        #     img_name = os.path.splitext(os.path.basename(self.img_files[index]))[0]
        #     # img_id = random.randint(1,5000)
        #     img_save_pathname = f"/home/lab/ckq/DOTA_split/plots/{img_name}.jpg"
        #     cv2.imwrite(img_save_pathname, img1)


        # labels_out: batch_id, cls_id, x,y,w,h,theta(其中x、y、w、h是归一化值)
        labels_out = torch.zeros((nl,7))
        if nl:
            # 4-points to [x,y,w,h,theta]
            

            # labels_rotate_boxes = points2rotation_boxes(labels[:,1:])
            labels_rotate_boxes = poly_to_rotated_box_np(labels[:, 1:])

            # if np.any(labels_rotate_boxes[:,-1] <=-45):
            # # if not (np.all(labels_rotate_boxes[:,-1] <=90) and np.all(labels_rotate_boxes[:,-1]) >0):
            #     print(self.img_files[index])
            
            # boxes_points = rotation_boxes2points(labels_rotate_boxes)
            # img2 = plot_rotate_boxes(img.copy(), boxes_points)
            # img_save_pathname = img_save_pathname.replace("plots", "plots2", 1)
            # cv2.imwrite(img_save_pathname, img2)

            # 坐标值归一化
            labels_rotate_boxes = xywhtheta2xywhtheta_n(labels_rotate_boxes, w=img.shape[1], h=img.shape[0])
        
            # 边界框参数
            labels_out[:, 2:] = torch.from_numpy(labels_rotate_boxes)
            # 类别参数
            labels_out[:, 1] = torch.from_numpy(labels[:,0])

        # Convert
        
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        # img = img.astype(np.float32)
        # img = img_normalize(img)
        
        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    # 暂时不用quad
    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                lb = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                lb = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(lb)

        for i, lb in enumerate(label4):
            lb[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, i):
    # loads 1 image from dataset index 'i', returns im, original hw, resized hw
    im = self.imgs[i]
    if im is None:  # not cached in ram
        npy = self.img_npy[i]
        if npy and npy.exists():  # load npy
            im = np.load(npy)
        else:  # read image
            path = self.img_files[i]
            im = cv2.imread(path)  # BGR
            assert im is not None, f'Image Not Found {path}'
        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
    else:
        return self.imgs[i], self.img_hw0[i], self.img_hw[i]  # im, hw_original, hw_resized


def load_mosaic(self, index):
    # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
    labels4 = []
    s = self.img_size
    yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    random.shuffle(indices)
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels = self.labels[index].copy()
        
        if labels.size:
            # 从归一化值，转化为以像素为单位，同时坐标值转化为在拼接图像上的坐标
            labels[:, 1:] = x1y1x2y2x3y3x4y4n2x1y1x2y2x3y3x4y4(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
        labels4.append(labels)

   
    # # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
     # 这个地方怎么办？旋转框角点需要截断吗？？？
    # 还是先不截断了
    # # 我们这里的做法是，截断，因为在进行mosaic时，有可能一个斜长的框的两个角点都在拼接图像范围之外了
    # for x in labels4[:, 1:]:
    #     np.clip(x, 0, 2*s-1, out=x)  
    # # # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    # 复制-粘贴数据增强，我们这里先不使用了
    # img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
    img4, labels4 = random_perspective_rotation(img4, labels4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img4, labels4


def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def verify_image_label(args):
    # Verify one image-label pair
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg = 0, 0, 0, 0, ''  # number (missing, found, empty, corrupt), message
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING: {im_file}: corrupt JPEG restored and saved'

        # 验证旋转框的标签是否合规
        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                # [cls, x1,y1, x2,y2, x3,y3, x4,y4]
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                # 标签信息，必须有9列数值，类别标签+8个坐标值
                assert lb.shape[1] == 9, f'labels require 9 columns, {lb.shape[1]} columns detected'
                
                # 下面这两个条件可以不满足，因为旋转框角点坐标可能会超出图像边界，可能会小于0，也可能会大于1
                # 但我们这里先不注释掉了
                # 所有的标注信息，都得大于0，也就是旋转框坐标值必须大于0
                assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                # 所有的坐标值必须小于1，也就是说，坐标值根据图像的宽高进行归一化
                assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                
                # 判断是否有重复的标注框
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates

                    msg = f'{prefix}WARNING: {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 9), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 9), dtype=np.float32)
        
        return im_file, lb, shape, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, nm, nf, ne, nc, msg]


def plot_rotate_boxes(img, boxes_points, cls_fall_point=0, color=(0, 0, 255), thickness=1):
    '''
        画旋转框，以及目标类别
    inputs:
        img:
        box_points: shape:[N,9] or [N,8], np.float64类型
        classes_name:
        cls_fall_point:表示，目标类别文本画在旋转框四个角点的哪个角上
    return:
        img
    '''
    num_cols = boxes_points.shape[1]
    # 逐个画box
    for box_points in boxes_points:
        if num_cols == 9:
            # 目标类别
            cls_id = int(box_points[0])
            box_points = box_points[1:].reshape(4,2).astype(np.int32)
        elif num_cols == 8:
            box_points = box_points.reshape(4,2).astype(np.int32)
        # 画4个点组成的轮廓，也就是旋转框
        # cv2.drawContours(image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None)
        # contours:轮廓列表，每个元素都是一个轮廓, 元素可以是一个numpy数组，
        # contourIdx：表示画列表中的哪个轮廓，-1表示画所有轮廓
        # thickness：如果是-1（cv2.FILLED），则为填充模式。
        cv2.drawContours(img, [box_points], contourIdx=-1, color=color, thickness=thickness)
        
        # if cls_fall_point <0 or cls_fall_point > 3:
        #     print(f"cls_fall_pont must be 0-3 !!!")
        #     exit()
        # pt1 = tuple(box_points[cls_fall_point])
        # # cv2.putText，按照顺序，各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
        # cv2.putText(img, 
        #     text = classes_name[cls_id], 
        #     org = pt1, 
        #     fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
        #     fontScale = 1, 
        #     color = color, 
        #     thickness = thickness
        # )

    return img




# # 标准化操作
# 图像标准化操作
def img_normalize(img):
    # img_normalize_mean =  np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(3,1,1)
    # img_normalize_std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(3,1,1)
    # img = (img - img_normalize_mean) / img_normalize_std

    img_normalize_mean =  [123.675, 116.28, 103.53]
    img_normalize_std = [58.395, 57.12, 57.375]

    img[0,:,:] = (img[0,:,:] - img_normalize_mean[0]) / img_normalize_std[0]
    img[1,:,:] = (img[1,:,:] - img_normalize_mean[1]) / img_normalize_std[1]
    img[2,:,:] = (img[2,:,:] - img_normalize_mean[2]) / img_normalize_std[2]

    return img

def img_denormalize(img):
    
    img_normalize_mean =  [123.675, 116.28, 103.53]
    img_normalize_std = [58.395, 57.12, 57.375]

    img[0,:,:] = (img[0,:,:] * img_normalize_std[0]) + img_normalize_mean[0]
    img[1,:,:] = (img[1,:,:] * img_normalize_std[1]) + img_normalize_mean[1]
    img[2,:,:] = (img[2,:,:] * img_normalize_std[2]) + img_normalize_mean[2]
    
    img = img.clip(0.0,255.0)

    return img

def img_batch_normalize(img):
    # img_normalize_mean =  np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(3,1,1)
    # img_normalize_std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(3,1,1)
    # img = (img - img_normalize_mean) / img_normalize_std

    img_normalize_mean =  [123.675, 116.28, 103.53]
    img_normalize_std = [58.395, 57.12, 57.375]

    img[:,0,:,:] = (img[:,0,:,:] - img_normalize_mean[0]) / img_normalize_std[0]
    img[:,1,:,:] = (img[:,1,:,:] - img_normalize_mean[1]) / img_normalize_std[1]
    img[:,2,:,:] = (img[:,2,:,:] - img_normalize_mean[2]) / img_normalize_std[2]

    return img


def img_batch_denormalize(img):
    
    img_normalize_mean =  [123.675, 116.28, 103.53]
    img_normalize_std = [58.395, 57.12, 57.375]

    img[:,0,:,:] = (img[:,0,:,:] * img_normalize_std[0]) + img_normalize_mean[0]
    img[:,1,:,:] = (img[:,1,:,:] * img_normalize_std[1]) + img_normalize_mean[1]
    img[:,2,:,:] = (img[:,2,:,:] * img_normalize_std[2]) + img_normalize_mean[2]
    
    img = img.clip(0.0,255.0)

    return img