import os
import os.path as osp

from ImgSplit_multi_process import splitbase as splitbase_trainval

def mkdir_if_not_exists(path):
    if not osp.exists(path):
        os.mkdir(path)


def prepare_multi_scale_data(src_path, dst_path, gap=200, subsize=1024, scales=[0.5, 1.0, 1.5], num_process=32): 
    # make dst path if not exist
    mkdir_if_not_exists(dst_path)
    dst_train_path = osp.join(dst_path, 'train')
    dst_val_path = osp.join(dst_path, 'val')


    # split train data
    print('split train data')
    src_train_path = osp.join(src_path, 'train')
    split_train = splitbase_trainval(src_train_path, dst_train_path,
                                     gap=gap, subsize=subsize, num_process=num_process)
    # 开始正式切图像
    for scale in scales:
        split_train.splitdata(scale)


    print('split val data')
    # split val data
    src_val_path = osp.join(src_path, 'val')
    split_val = splitbase_trainval(src_val_path, dst_val_path,
                                   gap=gap, subsize=subsize, num_process=num_process)
    for scale in scales:
        split_val.splitdata(scale)


    print('done!')


if __name__ == '__main__':
    
    # DOTA数据集的路径
    data_path = "/home/lab/ckq/DOTA/"
    # 切片后的保存路径
    data_save_path = "/home/lab/ckq/DOTA_split/"

    prepare_multi_scale_data(data_path, data_save_path, gap=200, subsize=1024, scales=[1.0],
                             num_process=32)
