import os.path as osp
import os
import shutil
import numpy as np

import cv2
import tqdm

classes_name = ('vehicle', )

# 将DOTA的组织格式，转化为yolo的格式，
def convert_dota_to_yolo(base_path, subdataset):
    
    # yolo格式标签的存储位置
    yolo_txt_path = osp.join(base_path, subdataset, "labels_hbbox")

    if not osp.exists(yolo_txt_path):
        os.mkdir(yolo_txt_path)

    # DOTA的labelTxt格式的标签路径
    label_path = osp.join(base_path, subdataset, "labels")

    label_ls = sorted(os.listdir(label_path))
    # 把标签格式，组织为yolo的形式，[cls_id, x1,y1, x2,y2, x3,y3, x4,y4]
    # 并且根据图像的宽高把坐标值进行归一化
    for labelTxt_name in tqdm.tqdm(label_ls):

        labelTxt_pathname = osp.join(label_path, labelTxt_name)
        with open(labelTxt_pathname, 'r') as f:
            lines = f.readlines()
        # 去掉换行符号
        lines = [line.strip('\n') for line in lines]
        
        lines = [line.split(' ') for line in lines]


        # 创建yolo格式的txt标注文件
        new_lines = []
        for line in lines:
            cls_id = line[0]
            # 类别标签放在第一位置
            new_line = [cls_id]

            box_points = np.array(list(map(float, line[1:]))).reshape(4,2)
            xmin = box_points[:,0].min()
            xmax = box_points[:,0].max()
            ymin = box_points[:,1].min()
            ymax = box_points[:,1].max()
            hbbox_w = xmax - xmin
            hbbox_h = ymax - ymin
            x_c = xmin + hbbox_w / 2
            y_c = ymin + hbbox_h / 2
            new_line = new_line + [str(x_c), str(y_c), str(hbbox_w), str(hbbox_h)]
            
            new_lines.append(new_line)
        

        new_lines = [' '.join(line) for line in new_lines]
        new_lines = [line+"\n" for line in new_lines]

        save_txt_pathname = osp.join(yolo_txt_path , labelTxt_name)
        with open(save_txt_pathname, 'w') as f:
            f.writelines(new_lines)




if __name__=="__main__":
    DOTA_path = "/home/lab/ckq/LARVehicle/LAR1024/"
    
    # 被过滤掉的困难样本
    fliter_difficult_tuple = ('2',)

    subdataset = "train"
    convert_dota_to_yolo(DOTA_path, subdataset)

    # # 验证的时候不能过滤掉无目标的图像
    # subdataset = "val"
    # convert_dota_to_yolo(DOTA_path, subdataset, filter_empty_img=False, fliter_difficult_tuple=fliter_difficult_tuple)

    # 验证的时候不能过滤掉无目标的图像
    subdataset = "test"
    convert_dota_to_yolo(DOTA_path, subdataset)
