import os
import os.path as osp

from DOTA_devkit.ImgSplit_multi_process import splitbase as splitbase_trainval
from DOTA_devkit.SplitOnlyImage_multi_process import splitbase as splitbase_test
from DOTA_devkit.convert_dota_to_mmdet import convert_dota_to_mmdet


def mkdir_if_not_exists(path):
    if not osp.exists(path):
        os.mkdir(path)


def prepare_multi_scale_data(src_path, dst_path, gap=200, subsize=1024, scales=[0.5, 1.0, 1.5], num_process=32):
    dst_train_path = osp.join(dst_path, 'train_split')
    dst_val_path = osp.join(dst_path, 'val_split')
    dst_test_path = osp.join(dst_path, 'test_split')
    # make dst path if not exist
    mkdir_if_not_exists(dst_path)
    mkdir_if_not_exists(dst_train_path)
    mkdir_if_not_exists(dst_val_path)
    mkdir_if_not_exists(dst_test_path)

    # split train data
    print('split train data')
    split_train = splitbase_trainval(osp.join(src_path, 'train'), dst_train_path,
                                     gap=gap, subsize=subsize, num_process=num_process)
    # 开始正式切图像
    for scale in scales:
        split_train.splitdata(scale)
    
    # trainval模式，会切割标签，filter_empty_gt会把无目标的图像切片过滤掉
    convert_dota_to_mmdet(dst_train_path, osp.join(dst_train_path, 'train1024.pkl'), 
                            trainval=True, filter_empty_gt=True)

    
    print('split val data')
    # split val data
    split_val = splitbase_trainval(osp.join(src_path, 'val'), dst_val_path,
                                   gap=gap, subsize=subsize, num_process=num_process)
    for scale in scales:
        split_val.splitdata(scale)
    convert_dota_to_mmdet(dst_val_path, osp.join(dst_val_path, 'val1024.pkl'), 
                            trainval=True, filter_empty_gt=False)



    # split test data
    print('split test data')
    split_test = splitbase_trainval(osp.join(src_path, 'test'), dst_test_path,
                                gap=gap, subsize=subsize, num_process=num_process)
    for scale in scales:
        split_test.splitdata(scale)

    convert_dota_to_mmdet(dst_test_path, osp.join(dst_test_path, 'test1024.pkl'), 
                            trainval=True, filter_empty_gt=False)
    print('done!')


if __name__ == '__main__':
    # prepare_multi_scale_data('/data/hjm/dota', '/data/hjm/dota_1024', gap=200, subsize=1024, scales=[1.0],
    #                          num_process=32)

    src_dir = "/home/lab/ckq/LARVehicle/dota_style"
    dst_dir = "/home/lab/ckq/LARVehicle/LARVehicle_1024"
    prepare_multi_scale_data(src_dir, dst_dir, gap=320, subsize=1024, scales=[1.0],
                             num_process=32)
