import os
import os.path as osp

from DOTA_devkit.ImgSplit_multi_process import splitbase as splitbase_trainval
# from DOTA_devkit.SplitOnlyImage_multi_process import splitbase as splitbase_test
# from DOTA_devkit.convert_dota_to_mmdet import convert_dota_to_mmdet
# 

def mkdir_if_not_exists(path):
    if not osp.exists(path):
        os.mkdir(path)


def prepare_multi_scale_data(src_path, dst_path, gap=200, subsize=1024, scales=[0.5, 1.0, 1.5], num_process=32):

    # make dst path if not exist
    mkdir_if_not_exists(dst_path)

    # split train data
    print('split train data')
    split_train = splitbase_trainval(src_path, dst_path,
                                     gap=gap, subsize=subsize, num_process=num_process)
    # 开始正式切图像
    for scale in scales:
        split_train.splitdata(scale)
    
    
    print('done!')


if __name__ == '__main__':
    # prepare_multi_scale_data('/data/hjm/dota', '/data/hjm/dota_1024', gap=200, subsize=1024, scales=[1.0],
    #                          num_process=32)

    src_dir = "/home/lab/ckq/LARVehicle/choosed_dataset/labels_cleaning/total_save_remove_some_labels_split/"
    dst_dir = "/home/lab/ckq/LARVehicle/choosed_dataset/labels_cleaning/total_save_remove_some_labels_split/split/"
    prepare_multi_scale_data(src_dir, dst_dir, gap=320, subsize=1024, scales=[1.0],
                             num_process=32)
