import os.path as osp
import os
import shutil
import numpy as np

import cv2
import tqdm

classes_name = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
                'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
                'harbor', 'swimming-pool', 'helicopter')

# 将DOTA的组织格式，转化为yolo的格式，
def convert_dota_to_yolo(base_path, subdataset, filter_empty_img=True, fliter_difficult_tuple=()):
    
    # yolo格式标签的存储位置
    yolo_txt_path = osp.join(base_path, subdataset, "labels")

    if not osp.exists(yolo_txt_path):
        os.mkdir(yolo_txt_path)

    # DOTA的labelTxt格式的标签路径
    labelTxt_path = osp.join(base_path, subdataset, "labelTxt")
    imgs_path = osp.join(base_path, subdataset, "images")

    labelTxt_ls = sorted(os.listdir(labelTxt_path))
    # 把标签格式，组织为yolo的形式，[cls_id, x1,y1, x2,y2, x3,y3, x4,y4]
    # 并且根据图像的宽高把坐标值进行归一化
    for labelTxt_name in tqdm.tqdm(labelTxt_ls):

        labelTxt_pathname = osp.join(labelTxt_path, labelTxt_name)
        with open(labelTxt_pathname, 'r') as f:
            lines = f.readlines()
        # 去掉换行符号
        lines = [line.strip('\n') for line in lines]
        
        lines = [line.split(' ') for line in lines]
        # 注意，过滤掉困难样本后，可能部分有目标的图像就变成无目标图像了
        if len(fliter_difficult_tuple):
            # 原始的difficult标签为0、1，切图后增加了2标签，我们只保留0目标
            lines = [line[:-1] for line in lines if line[-1] not in fliter_difficult_tuple]
        else:
            lines = [line[:-1] for line in lines]

        # 创建yolo格式的txt标注文件
        img_name = labelTxt_name.replace(".txt", ".png", 1)
        img_pathname = osp.join(imgs_path, img_name)
        img_h, img_w, _ = cv2.imread(img_pathname).shape
        new_lines = []
        for line in lines:
            cls_id = classes_name.index(line[-1])
            # 类别标签放在第一位置
            new_line = [str(cls_id)]
            box_points = np.array(list(map(float, line[:-1]))).reshape(-1,8)
            box_points[:,[0,2,4,6]] /= img_w
            box_points[:,[1,3,5,7]] /= img_h
            new_line = new_line + list(map(str, box_points.reshape(-1).tolist()))
            
            new_lines.append(new_line)
        

        new_lines = [' '.join(line) for line in new_lines]
        new_lines = [line+"\n" for line in new_lines]

        save_txt_pathname = osp.join(yolo_txt_path , labelTxt_name)
        with open(save_txt_pathname, 'w') as f:
            f.writelines(new_lines)



    # 如果过滤掉空图像，把空标签和空图像移动出来
    if filter_empty_img:

        imgs_empty_move_path = imgs_path.replace("images", "empty_images", 1)
        yolo_txt_empty_move_path = yolo_txt_path.replace("labels", "empty_labels", 1)

        for i_path in (imgs_empty_move_path, yolo_txt_empty_move_path):
            if not os.path.exists(i_path):
                os.mkdir(i_path)

        # 空文件列表
        yolo_txt_empty_ls = []
        yolo_txt_ls = sorted(os.listdir(yolo_txt_path))
        for yolo_txt_name in yolo_txt_ls:
            
            yolo_txt_pathname = osp.join(yolo_txt_path, yolo_txt_name)
            with open(yolo_txt_pathname, 'r') as f:
                lines = f.readlines()
            
            # 如果不存在标注框
            if len(lines) == 0:
                yolo_txt_empty_ls.append(yolo_txt_name)

                # 把空标签和空图像移动出来
                shutil.move(yolo_txt_pathname, yolo_txt_empty_move_path)

                img_pathname = osp.join(imgs_path , yolo_txt_name.replace(".txt", ".png", 1))
                shutil.move(img_pathname, imgs_empty_move_path)


        num_empty = len(yolo_txt_empty_ls)
        print(f"空文件个数：{num_empty}")
    

if __name__=="__main__":
    DOTA_path = "/home/lab/ckq/DOTA_split/"
    subdataset = "train"
    # 被过滤掉的困难样本
    fliter_difficult_tuple = ('1','2',)

    convert_dota_to_yolo(DOTA_path, subdataset, filter_empty_img=True, fliter_difficult_tuple=fliter_difficult_tuple)

    # 验证的时候不能过滤掉无目标的图像
    subdataset = "val"
    convert_dota_to_yolo(DOTA_path, subdataset, filter_empty_img=False, fliter_difficult_tuple=fliter_difficult_tuple)
