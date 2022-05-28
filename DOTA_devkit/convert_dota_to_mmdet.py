import os
import os.path as osp

import mmcv
import numpy as np
from PIL import Image

# from mmdet.core import poly_to_rotated_box_single
from utils.general import poly_to_rotated_box_single

classes_name = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
               'harbor', 'swimming-pool', 'helicopter']


label_ids = {name: i + 1 for i, name in enumerate(classes_name)}


def parse_ann_info(label_base_path, img_name):
    lab_path = osp.join(label_base_path, img_name + '.txt')
    bboxes, labels, bboxes_ignore, labels_ignore = [], [], [], []
    with open(lab_path, 'r') as f:
        for ann_line in f.readlines():
            ann_line = ann_line.strip().split(' ')
            bbox = [float(ann_line[i]) for i in range(8)]
            # 8 point to 5 point xywha
            bbox = tuple(poly_to_rotated_box_single(bbox).tolist())
            class_name = ann_line[8]
            difficult = int(ann_line[9])
            
            # 忽略困难标签为2的边界框（这种边界框是自己切图时引入的）
            # ignore difficult =2
            if difficult == 0:
                bboxes.append(bbox)
                labels.append(label_ids[class_name])
            elif difficult == 1:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label_ids[class_name])
    return bboxes, labels, bboxes_ignore, labels_ignore


def convert_dota_to_mmdet(src_path, out_path, trainval, filter_empty_gt, ext='.png'):
    """Generate .pkl format annotation that is consistent with mmdet.
    Args:
        src_path: dataset path containing images and labelTxt folders.
        out_path: output pkl file path
        trainval: trainval or test
    """
    img_path = os.path.join(src_path, 'images')
    label_path = os.path.join(src_path, 'labelTxt')
    img_lists = os.listdir(img_path)

    data_dict = []
    for id, img in enumerate(img_lists):
        img_info = {}
        img_name = osp.splitext(img)[0]
        label = os.path.join(label_path, img_name + '.txt')
        img = Image.open(osp.join(img_path, img))
        img_info['filename'] = img_name + ext
        img_info['height'] = img.height
        img_info['width'] = img.width
        if trainval:
            # 找不到对应的txt标签的，跳过
            # 这样不行，找不到txt的，直接报错退出
            if not os.path.exists(label):
                print('Label:' + img_name + '.txt' + ' Not Exist')
                exit()
                # continue

            # filter_empty_gt默认为True，也即是说，标签为空的（即没有目标的），就过滤掉
            # 这样也不行，负样本也是有用的，filter_empty_gt改为没有默认值，必须用户指定
            # filter images without gt to speed up training
            if filter_empty_gt & (osp.getsize(label) == 0):
                continue
            
            # 这个bboxes_ignore，就是DOTA中的difficult样本
            bboxes, labels, bboxes_ignore, labels_ignore = parse_ann_info(label_path, img_name)
            ann = {}
            ann['bboxes'] = np.array(bboxes, dtype=np.float32)
            ann['labels'] = np.array(labels, dtype=np.int64)
            ann['bboxes_ignore'] = np.array(bboxes_ignore, dtype=np.float32)
            ann['labels_ignore'] = np.array(labels_ignore, dtype=np.int64)
            img_info['ann'] = ann
        data_dict.append(img_info)

    mmcv.dump(data_dict, out_path)


if __name__ == '__main__':

    src_dir = "/home/lab/ckq/DOTA_split/"

    dst_train_path = osp.join(src_dir, 'train')
    # trainval模式，会切割标签，filter_empty_gt会把无目标的图像切片过滤掉
    convert_dota_to_mmdet(dst_train_path, osp.join(dst_train_path, 'train1024.pkl'), 
                            trainval=True, filter_empty_gt=True)

    dst_val_path = osp.join(src_dir, 'val')
    # trainval模式，会切割标签，filter_empty_gt会把无目标的图像切片过滤掉
    convert_dota_to_mmdet(dst_val_path, osp.join(dst_val_path, 'val1024.pkl'), 
                            trainval=True, filter_empty_gt=False)
    


    print('done!')
