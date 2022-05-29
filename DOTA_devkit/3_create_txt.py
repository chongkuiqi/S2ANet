# S2ANet原始代码中，在进行验证时，需要一个不带扩展名的图像名字列表

import os

base_dir = "/home/lab/ckq/DOTA_split/val/"

imgs_dir = base_dir + "images/"

txt_pathname = base_dir + "val_split.txt"

imgs_ls = os.listdir(imgs_dir)

lines=[]
for img_name in imgs_ls:
    # 去掉后缀名
    img_name = os.path.splitext(img_name)[0]

    img_name = img_name + "\n"

    lines.append(img_name)

with open(txt_pathname, 'w') as f:
    f.writelines(lines)
