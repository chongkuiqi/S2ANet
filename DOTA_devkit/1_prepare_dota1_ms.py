import os
import os.path as osp

from ImgSplit_multi_process import splitbase as splitbase_trainval
from SplitOnlyImage_multi_process import splitbase as splitbase_test
from convert_dota_to_mmdet import convert_dota_to_mmdet


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
    
    # 对标签进行处理，转化为mmdetection框架的格式
    # trainval模式，会切割标签，filter_empty_gt会把无目标的图像切片过滤掉
    convert_dota_to_mmdet(dst_train_path, osp.join(dst_train_path, 'train1024.pkl'), 
                            trainval=True, filter_empty_gt=True)


    print('split val data')
    # split val data
    src_val_path = osp.join(src_path, 'val')
    split_val = splitbase_trainval(src_val_path, dst_val_path,
                                   gap=gap, subsize=subsize, num_process=num_process)
    for scale in scales:
        split_val.splitdata(scale)

    convert_dota_to_mmdet(dst_val_path, osp.join(dst_val_path, 'val1024.pkl'), 
                            trainval=True, filter_empty_gt=False)


    print('done!')


if __name__ == '__main__':
    data_path = "/home/lab/ckq/DOTA/"
    # 切片后的保存路径
    data_save_path = "/home/lab/ckq/DOTA_split/"
    prepare_multi_scale_data(data_path, data_save_path, gap=200, subsize=1024, scales=[1.0],
                             num_process=32)
