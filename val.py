# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python path/to/val.py --weights yolov5s.pt                 # PyTorch
                                      yolov5s.torchscript        # TorchScript
                                      yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                      yolov5s.xml                # OpenVINO
                                      yolov5s.engine             # TensorRT
                                      yolov5s.mlmodel            # CoreML (MacOS-only)
                                      yolov5s_saved_model        # TensorFlow SavedModel
                                      yolov5s.pb                 # TensorFlow GraphDef
                                      yolov5s.tflite             # TensorFlow Lite
                                      yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from utils.datasets_rotation import img_batch_normalize

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# from models.common import DetectMultiBackend
from utils.datasets_rotation import create_dataloader
from utils.general import (LOGGER, check_dataset, check_img_size, check_yaml,
                           colorstr, increment_path, print_args)
# from utils.plots import  plot_val_study
from utils.torch_utils import select_device, time_sync

import os.path as osp
from utils.general import scale_coords_rotated, rotated_box_to_poly_single
from DOTA_devkit.ResultMerge_multi_process import mergebypoly
from DOTA_devkit.dota_evaluation_task1 import voc_eval


def save_per_class(imgs_results_ls, dst_raw_path, classes_names_id:dict):
    # print('Saving results to {}'.format(dst_raw_path))
    
    # t = time_sync()
    classes_names_lines = {class_name:[] for class_name in  classes_names_id.values()}
    # for img_name, det_bboxes, det_labels in tqdm(imgs_results_ls):
    for img_name, det_bboxes, det_labels in imgs_results_ls:
        for det_bbox, det_class_id in zip(det_bboxes, det_labels):
            # 经过验证，如果det_bboxes是torch.tensor类型，全部处理完大约需要11分钟
                # 如果是列表类型，全部处理完大约需要6分钟
                # 如果是numpy类型，全部处理完大约需要7分钟
                # det_bbox = det_bbox.reshape(6).tolist()
                # class_id = det_label.item()
            class_name = classes_names_id[det_class_id]
            score = det_bbox[5]
            bbox = rotated_box_to_poly_single(det_bbox[:5])
            line = '{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(
                osp.splitext(img_name)[0], score, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4],
                bbox[5], bbox[6], bbox[7])
            
            classes_names_lines[class_name].append(line)
    
    for class_name, lines in classes_names_lines.items():
        class_txt_pathname = osp.join(dst_raw_path, class_name + '.txt')
        with open(class_txt_pathname, 'w') as f:
            f.writelines(lines)

    # t = time_sync() - t
    # print(f"save txts time: {t}s") 

def merge_per_class(dst_raw_path, dst_merge_path):
    # t = time_sync()
    ## 合并，转化为切割前大图上的坐标
    print('Merge results to {}'.format(dst_merge_path))
    mergebypoly(dst_raw_path, dst_merge_path)
    # t = time_sync() - t
    # print(f"merge time: {t}s")

    print('save and merge has Done ! ')

# 把检测结果，按照类别进行存储，每个类别存储为一个.txt文件
def save_and_merge(imgs_results_ls, dst_raw_path, dst_merge_path, classes_names_id:dict):

    print('Saving results to {}'.format(dst_raw_path))
    
    t = time_sync()
    classes_names_lines = {class_name:[] for class_name in  classes_names_id.values()}
    for img_name, det_bboxes, det_labels in tqdm(imgs_results_ls):
        for det_bbox, det_class_id in zip(det_bboxes, det_labels):
            # 经过验证，如果det_bboxes是torch.tensor类型，全部处理完大约需要11分钟
                # 如果是列表类型，全部处理完大约需要6分钟
                # 如果是numpy类型，全部处理完大约需要7分钟
                # det_bbox = det_bbox.reshape(6).tolist()
                # class_id = det_label.item()
            class_name = classes_names_id[det_class_id]
            score = det_bbox[5]
            bbox = rotated_box_to_poly_single(det_bbox[:5])
            line = '{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(
                osp.splitext(img_name)[0], score, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4],
                bbox[5], bbox[6], bbox[7])
            
            classes_names_lines[class_name].append(line)
    
    for class_name, lines in classes_names_lines.items():
        class_txt_pathname = osp.join(dst_raw_path, class_name + '.txt')
        with open(class_txt_pathname, 'w') as f:
            f.writelines(lines)

    t = time_sync() - t
    print(f"save txts time: {t}s")
                    

    t = time_sync()
    ## 合并，转化为切割前大图上的坐标
    print('Merge results to {}'.format(dst_merge_path))
    mergebypoly(dst_raw_path, dst_merge_path)
    t = time_sync() - t
    print(f"merge time: {t}s")

    print('save and merge has Done ! ')

@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        save_txt=False,  # save results to *.txt
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        is_mAP_split=True,
        ):
    
    # model表示是在训练状态中
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model

        # 默认会使用半精度进行验证
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Data
        data = check_dataset(data)  # check

        # Load model
        # 如果是从官方代码训练得到的模型中导入参数
        if str(weights).endswith('.pth'):
            state_dict = torch.load(weights, map_location=device)["state_dict"]

            # print(state_dict.keys())
            # exit()

            # state_dict = torch.load(weights, map_location=device)["model"].state_dict()
            from models.s2anet import S2ANet
            num_classes = data['nc']
            model = S2ANet(num_classes=num_classes).to(device) # create

            def intersect_dicts(model_state_dict, pretrained_state_dict, exclude=()):
                assert len(model_state_dict.items()) == len(pretrained_state_dict.items())
                return {k1:v2 for (k1,v1),(k2,v2) in zip(model_state_dict.items(), pretrained_state_dict.items())}

            # # # print(state_dict)
            # for (k1,v1),(k2,v2) in zip(state_dict.items(), model.state_dict().items()):
            #     print(k1, ':', k2)
            # print(len(state_dict), ':', len(model.state_dict()))

            # BN层的num_batches_tracked参数不导入
            exclude = []
            state_dict = intersect_dicts(model.state_dict(), state_dict, exclude=exclude)

            # print(len(state_dict), ':', len(model.state_dict()))
            model.load_state_dict(state_dict, strict=True)

        else:
            model = torch.load(weights, map_location=device)["model"]
        

        print(model.head.iou_thres_nms)
        model.head.iou_thres_nms = 0.1
        print(model.head.iou_thres_nms)
        print(model)
        exit()

        # stride, pt, jit, onnx, engine = model.stride, model.pt, model.jit, model.onnx, model.engine
        stride, pt, jit, onnx, engine = model.stride, True, False, False, False
        stride = max(stride)
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if pt or jit:
            # model.model.half() if half else model.model.float()
            model.half() if half else model.float()
        elif engine:
            batch_size = model.batch_size
        else:
            half = False
            batch_size = 1  # export.py models default to batch-size 1
            device = torch.device('cpu')
            LOGGER.info(f'Forcing --batch-size 1 square inference shape(1,3,{imgsz},{imgsz}) for non-PyTorch backends')

        

    # Configure
    model.eval()
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    names = ['item'] if single_cls and len(data['names']) != 1 else data['names']  # class names
    if not hasattr(model, 'names'):
        model.names = names

    # Dataloader
    if not training:
        # model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz), half=half)  # warmup
        pad = 0.0 if task == 'speed' else 0.5
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, stride, single_cls, pad=pad, rect=pt,
                                       workers=workers, prefix=colorstr(f'{task}: '))[0]

    # 图像个数
    seen = 0

    classes_names_id = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    # s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    s = ('%20s' + '%11s' * 5) % ('Class', 'P', 'R', 'mAP@.5', 'mF1', 'conf')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    # 设置为numpy，减少显存占用
    # loss_items_sum = torch.zeros(4, device=device)
    loss_items_sum = np.zeros(4)
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    
    # 存储整个数据集的推理结果
    imgs_results_ls_save = []
    for batch_i, (im, targets, paths, shapes0) in enumerate(pbar):
        
        # if batch_i > 1:
        #     break
        # 前处理的时间
        t1 = time_sync()
        if pt or jit or engine:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        # im /= 255  # 0 - 255 to 0.0 - 1.0
        im = img_batch_normalize(im)

        # nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference， 推理过程+后处理过程
        # out, train_out = model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs
        _, loss_items, imgs_results_ls = model(im, targets, post_process=True)

        # k,v = next(model.named_parameters())
        # print(v.dtype)
        # exit()
        dt[1] += time_sync() - t2

        # 损失累加
        loss_items_sum += loss_items

        ## 下面对网络边界框解码后的结果进行处理
        # 网络输出的边界框，单位是像素，是相对于输入图像尺寸的，需要调整为相对原始图像尺寸的
        for img_id, img_results in enumerate(imgs_results_ls):
            # seen += 1
            # det_bboxes shape : [N, 6(x,y,w,h,theta,score)], labels shape:[N,1]
            det_bboxes, det_labels = img_results
            img_pathname, shape0, ratio_pad = Path(paths[img_id]), shapes0[img_id][0], shapes0[img_id][1]
            
            # 输入图像尺寸上的边界框坐标，转化为原始图像尺寸上的坐标
            scale_coords_rotated(im[img_id].shape[1:], det_bboxes, shape0, ratio_pad)
            img_name = os.path.basename(img_pathname)
            det_bboxes = det_bboxes.reshape(-1,6).tolist()
            det_labels = det_labels.reshape(-1).tolist()
            # det_bboxes = det_bboxes.reshape(-1,6).cpu().numpy()
            # det_labels = det_labels.reshape(-1).cpu().numpy()
            imgs_results_ls_save.append((img_name, det_bboxes, det_labels))
        
    

    ## 按类别储存为.txt文件，并进行合并，转化为切割前大图上的坐标
    dst_raw_path = osp.join(save_dir, 'results_before_nms')
    dst_merge_path = osp.join(save_dir, 'results_after_nms')
    # dst_raw_path = osp.join("/home/lab/ckq/yolov5_new/runs/train/exp37", 'results_before_nms')
    # dst_merge_path = osp.join("/home/lab/ckq/yolov5_new/runs/train/exp37", 'results_after_nms')
    for path in (dst_raw_path, dst_merge_path):
        if not osp.exists(path):
            os.mkdir(path)
    
    
    save_per_class(imgs_results_ls_save, dst_raw_path, classes_names_id)
    if is_mAP_split:

        ## 下面计算精度指标mAP
        # NMS后检测结果的存储路径，每个类别的检测结果单独存储为一个文件
        detpath = osp.join(dst_raw_path, '{:s}.txt')
        # gt 标签的存储路径，每个图像文件的标签单独存储为一个文件
        if task == "val":
            gt_dir = data["val_split_imgs_gt_path"]
            imagesetfile = data["val_split_imgs_ls_txt_path"]
        elif task == "test":
            gt_dir = data["test_split_imgs_gt_path"]
            imagesetfile = data["test_split_imgs_ls_txt_path"]
        else:
            print("错误")
            exit()
    else:

        merge_per_class(dst_raw_path, dst_merge_path)
        ## 下面计算精度指标mAP
        # NMS后检测结果的存储路径，每个类别的检测结果单独存储为一个文件
        detpath = osp.join(dst_merge_path, '{:s}.txt')
        # gt 标签的存储路径，每个图像文件的标签单独存储为一个文件
        if task == "val":
            gt_dir = data["val_complete_imgs_gt_path"]
            imagesetfile = data["val_complete_imgs_ls_txt_path"]
        elif task == "test":
            gt_dir = data["test_complete_imgs_gt_path"]
            imagesetfile = data["test_complete_imgs_ls_txt_path"]
        else:
            print("错误")
            exit()

    
    annopath = osp.join(gt_dir, '{:s}.txt')
    ## 逐个类别计算AP精度
    classes_ap50s = []
    classes_P = []
    classes_R = []
    classes_f1 = []
    classes_conf = []
    classes_num_det = []

    # t = time_sync()
    for class_id, class_name in classes_names_id.items():
        # 这个置信度sorted_scores是从大到小排序的
        # detpath: 检测结果的存储路径，检测结果的边界框是切割前的图像上的坐标
        # annopath：测试集或验证集的标签文件的存储路径，边界框是切割前的图像上的坐标
        # imagesetfile：测试集或验证集图像的列表，不包括拓展名
        recall, precsion, ap50, sorted_scores = voc_eval(detpath,
                                    annopath,
                                    imagesetfile,
                                    class_name,
                                    is_filter_difficult=True,
                                    ovthresh=0.5,
                                    use_07_metric=True)
        # print(classname, ': ', ap50)
        classes_ap50s.append(ap50)
        

        f1 = 2 * recall * precsion / (recall + precsion + 1e-16)
        
        # print(f"{classname} f1:{f1[-1]}")
        # 找到使f1最大的置信度
        max_f1_idx = f1.argmax()
        # 找到最后的置信度阈值，获得最大的召回
        # max_f1_idx = f1.size - 1

        max_f1 = f1[max_f1_idx]
        max_f1_conf = sorted_scores[max_f1_idx]
        num_det = max_f1_idx+1
        
        classes_P.append(precsion[max_f1_idx])
        classes_R.append(recall[max_f1_idx])
        classes_f1.append(max_f1)
        classes_conf.append(max_f1_conf)
        classes_num_det.append(num_det)
        # print(f'class:{class_name}, num_dets:{num_det}, conf:{max_f1_conf}, p:{precsion[max_f1_idx]}, r:{recall[max_f1_idx]}, f1:{max_f1}')

    classes_ap50s = np.array(classes_ap50s)
    classes_P = np.array(classes_P)
    classes_R = np.array(classes_R)
    classes_f1 = np.array(classes_f1)
    classes_conf = np.array(classes_conf)

    map50 = classes_ap50s.mean().item()
    mf1 = classes_f1.mean().item()
    mp = classes_P.mean().item()
    mr = classes_R.mean().item()
    conf = classes_conf.mean().item()



    # Print results
    # pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    # LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, mf1))
    pf = '%20s' + '%11.3g' * 5  # print format
    LOGGER.info(pf % ('all', mp, mr, map50, mf1, conf))

    # print(classes_ap50s)


    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    
    
    # maps = np.zeros(nc) + map
    # for i, c in enumerate(ap_class):
    #     maps[c] = ap[i]
    # return (mp, mr, map50, map, *(loss_items_sum.cpu() / len(dataloader)).tolist()), maps, t
    return (mp, mr, map50, conf, *(loss_items_sum / len(dataloader)).tolist()), classes_ap50s


def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data', type=str, default=ROOT / 'data/dota.yaml', help='dataset.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/LAR1024.yaml', help='dataset.yaml path')
    # parser.add_argument('--data', type=str, default=ROOT / 'data/dota_map.yaml', help='dataset.yaml path')

    # parser.add_argument('--weights', type=str, default=ROOT / 'runs/train/exp40/weights/best.pt', help='initial weights path')
    parser.add_argument('--weights', type=str, default=ROOT / 'runs/train/exp233/weights/best.pt', help='initial weights path')
    # parser.add_argument('--weights', type=str, default='/home/lab/ckq/S2ANet_offical/work_dirs/s2anet_r50_fpn_1x_ms_rotate_vehicle/exp9/epoch_3.pth', help='initial weights path')
    # parser.add_argument('--weights', type=str, default='/home/lab/ckq/S2ANet_offical/work_dirs/s2anet_r50_fpn_1x_dota/exp1/epoch_11.pth', help='initial weights path')

    # 是否以切图的方式计算mAP
    parser.add_argument('--is_mAP_split', action='store_true', default=True)

    parser.add_argument('--half', action='store_true', default=True,  help='use FP16 half-precision inference')

    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=1024, help='inference size (pixels)')

    parser.add_argument('--task', default='val', help='train, val, test')



    parser.add_argument('--device', default='3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')

    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    
    opt.data = check_yaml(opt.data)  # check YAML
    print_args(FILE.stem, opt)
    return opt


def main(opt):

    assert opt.task in ('train', 'val', 'test')  # run normally

    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
