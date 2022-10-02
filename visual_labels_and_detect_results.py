# 对./labels中的标签进行可视化

import cv2
import os
import shutil
import numpy as np
import tqdm

from utils_plot import load_DOTA_label, plot_rotate_boxes

def read_det_results(results_txt, conf_thr = 0.37):

    with open(results_txt, 'r') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]

    # 存储每个图像的边界框
    det_results = {}
    for line in lines:
        line = line.strip().split(' ')
        if len(line) == 10:

            img_name = line[0]
            conf = float(line[1])
            if conf > conf_thr:
                box_points = line[2:]
                box_points = list(map(float, box_points))

                if img_name not in det_results.keys():
                    det_results[img_name] = []
                
                det_results[img_name].append(box_points)
        else:
            print("错误")

    for k,v in det_results.items():
        det_results[k] = np.array(v).reshape(-1,8)
    
    return det_results


base_dir = "./LAR1024/test/"
imgs_path = base_dir + "images/"
txt_path = base_dir + "labelTxt/"

detection_results_txt = "/home/lab/ckq/S2ANet/runs/val/exp0/results_before_nms/vehicle.txt"


# the path to save images
visual_label_dir = base_dir + "visual/"

if not os.path.exists(visual_label_dir):
    os.mkdir(visual_label_dir)

# 检测的结果
det_results = read_det_results(detection_results_txt)
# # 漏掉的gt
# miss_gt_result = read_det_results(miss_gt_results_txt)

txt_ls = sorted(os.listdir(txt_path))
for txt_name in tqdm.tqdm(txt_ls):

    img_name = txt_name.replace(".txt", ".png", 1)
    img_pathname = imgs_path + img_name
    img = cv2.imread(img_pathname)

    # # plot gt boxes 
    # boxes_points, boxes_classname, boxes_difficult = load_DOTA_label(txt_pathname)
    # if boxes_points is not None:
    #     img = plot_rotate_boxes(img, boxes_points, color=(0,255,0))
    
    # plot detection results
    img_name_wo_ext = os.path.splitext(img_name)[0]
    if img_name_wo_ext in det_results.keys():
        boxes_points_det = det_results[img_name_wo_ext]
        img = plot_rotate_boxes(img, boxes_points_det, color=(0,0,255), thickness=3)



    img_save_name = img_name.replace(".png", ".jpg", 1)

    save_name = visual_label_dir + img_save_name

    cv2.imwrite(save_name, img)

