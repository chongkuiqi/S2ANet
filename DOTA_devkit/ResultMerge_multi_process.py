"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 15 result files, for the format, you can refer to "http://captain.whu.edu.cn/DOTAweb/tasks.html"
    search for PATH_TO_BE_CONFIGURED to config the paths
    Note, the evaluation is on the large scale images
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, '..')
import DOTA_devkit.dota_utils as util
import DOTA_devkit.polyiou.polyiou as polyiou
import pdb
import math
from multiprocessing import Pool
from functools import partial

# ## the thresh for nms when merge image
# nms_thresh = 0.5


# def py_cpu_nms_poly(dets, thresh=0.5):
#     scores = dets[:, 8]
#     polys = []
#     areas = []
#     for i in range(len(dets)):
#         tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
#                                            dets[i][2], dets[i][3],
#                                            dets[i][4], dets[i][5],
#                                            dets[i][6], dets[i][7]])
#         polys.append(tm_polygon)
#     order = scores.argsort()[::-1]

#     keep = []
#     while order.size > 0:
#         ovr = []
#         i = order[0]
#         keep.append(i)
#         for j in range(order.size - 1):
#             iou = polyiou.iou_poly(polys[i], polys[order[j + 1]])
#             ovr.append(iou)
#         ovr = np.array(ovr)

#         # print('ovr: ', ovr)
#         # print('thresh: ', thresh)
#         try:
#             if math.isnan(ovr[0]):
#                 pdb.set_trace()
#         except:
#             pass
#         inds = np.where(ovr <= thresh)[0]
#         # print('inds: ', inds)

#         order = order[inds + 1]

#     return keep


def py_cpu_nms_poly_fast(dets, thresh=0.5):
    obbs = dets[:, 0:-1]
    x1 = np.min(obbs[:, 0::2], axis=1)
    y1 = np.min(obbs[:, 1::2], axis=1)
    x2 = np.max(obbs[:, 0::2], axis=1)
    y2 = np.max(obbs[:, 1::2], axis=1)
    scores = dets[:, 8]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                           dets[i][2], dets[i][3],
                                           dets[i][4], dets[i][5],
                                           dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        # if order.size == 0:
        #     break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # w = np.maximum(0.0, xx2 - xx1 + 1)
        # h = np.maximum(0.0, yy2 - yy1 + 1)
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        # h_keep_inds = np.where(hbb_ovr == 0)[0]
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou
            # ovr.append(iou)
            # ovr_index.append(tmp_order[j])

        # ovr = np.array(ovr)
        # ovr_index = np.array(ovr_index)
        # print('ovr: ', ovr)
        # print('thresh: ', thresh)
        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(hbb_ovr <= thresh)[0]

        # order_obb = ovr_index[inds]
        # print('inds: ', inds)
        # order_hbb = order[h_keep_inds + 1]
        order = order[inds + 1]
        # pdb.set_trace()
        # order = np.concatenate((order_obb, order_hbb), axis=0).astype(np.int)
    return keep


# def py_cpu_nms(dets, thresh=0.5):
#     """Pure Python NMS baseline."""
#     # print('dets:', dets)
#     x1 = dets[:, 0]
#     y1 = dets[:, 1]
#     x2 = dets[:, 2]
#     y2 = dets[:, 3]
#     scores = dets[:, 4]

#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     ## index for dets
#     order = scores.argsort()[::-1]

#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         yy2 = np.minimum(y2[i], y2[order[1:]])

#         w = np.maximum(0.0, xx2 - xx1 + 1)
#         h = np.maximum(0.0, yy2 - yy1 + 1)
#         inter = w * h
#         ovr = inter / (areas[i] + areas[order[1:]] - inter)

#         inds = np.where(ovr <= thresh)[0]
#         order = order[inds + 1]

#     return keep


def nmsbynamedict(nameboxdict, nms):
    nameboxnmsdict = {x: [] for x in nameboxdict}
    for imgname in nameboxdict:
        # print('imgname:', imgname)
        # keep = py_cpu_nms(np.array(nameboxdict[imgname]), thresh)
        # print('type nameboxdict:', type(nameboxnmsdict))
        # print('type imgname:', type(imgname))
        # print('type nms:', type(nms))
        keep = nms(np.array(nameboxdict[imgname]))
        # print('keep:', keep)
        outdets = []
        # print('nameboxdict[imgname]: ', nameboxnmsdict[imgname])
        for index in keep:
            # print('index:', index)
            outdets.append(nameboxdict[imgname][index])
        nameboxnmsdict[imgname] = outdets
    return nameboxnmsdict


def poly2origpoly(poly, x, y, rate):
    origpoly = []
    for i in range(int(len(poly) / 2)):
        tmp_x = float(poly[i * 2] + x) / float(rate)
        tmp_y = float(poly[i * 2 + 1] + y) / float(rate)
        origpoly.append(tmp_x)
        origpoly.append(tmp_y)
    return origpoly

# 合并单个文件的检测结果
def mergesingle(dstpath, nms, fullname):
    name = util.custombasename(fullname)
    # print('name:', name)
    dstname = os.path.join(dstpath, name + '.txt')
    # 不打印了
    # print(dstname)

    with open(fullname, 'r') as f_in:
        # 是个字典，存储每个图像的检测结果，{img_name:[[x1...y4,conf]]}
        nameboxdict = {}
        lines = f_in.readlines()
        splitlines = [x.strip().split(' ') for x in lines]
        for splitline in splitlines:
            subname = splitline[0]
            splitname = subname.split('__')
            # 根据图像切片的名字，找到大图的名字
            oriname = splitname[0]
            pattern1 = re.compile(r'__\d+___\d+')
            # print('subname:', subname)
            x_y = re.findall(pattern1, subname)
            x_y_2 = re.findall(r'\d+', x_y[0])
            x, y = int(x_y_2[0]), int(x_y_2[1])

            pattern2 = re.compile(r'__([\d+\.]+)__\d+___')

            rate = re.findall(pattern2, subname)[0]

            confidence = splitline[1]
            poly = list(map(float, splitline[2:]))
            origpoly = poly2origpoly(poly, x, y, rate)
            # det是个列表，9个参数（x1,y1,......x4,y4,conf)
            det = origpoly
            det.append(confidence)
            det = list(map(float, det))
            if (oriname not in nameboxdict):
                nameboxdict[oriname] = []
            nameboxdict[oriname].append(det)
        
        # 逐张图像进行NMS
        nameboxnmsdict = nmsbynamedict(nameboxdict, nms)
        # 把NMS后的结果写入该类别文件
        with open(dstname, 'w') as f_out:
            for imgname in nameboxnmsdict:
                for det in nameboxnmsdict[imgname]:
                    # print('det:', det)
                    confidence = det[-1]

                    # # 过滤掉低质量的框
                    # classes_conf_thres=(0.555604, 0.626835, 0.818774, 0.415922, 0.781585, 0.433823, 0.820298, 0.721398, 0.394541, 0.599741, 0.532775)
                    # CLASSES = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K')
                    # if confidence < classes_conf_thres[CLASSES.index(name)]:
                    #     print("跳过")
                    #     continue
                    
                    bbox = det[0:-1]
                    outline = imgname + ' ' + str(confidence) + ' ' + ' '.join(map(str, bbox))
                    # print('outline:', outline)
                    f_out.write(outline + '\n')


def mergebase_parallel(srcpath, dstpath, nms):
    pool = Pool(16)
    # 找到文件列表
    filelist = util.GetFileFromThisRootDir(srcpath)

    # 进行NMS 
    mergesingle_fn = partial(mergesingle, dstpath, nms)
    # pdb.set_trace()
    pool.map(mergesingle_fn, filelist)


def mergebase(srcpath, dstpath, nms):
    filelist = util.GetFileFromThisRootDir(srcpath)
    for filename in filelist:
        mergesingle(dstpath, nms, filename)


# 通过水平框NMS进行合并
def mergebyrec(srcpath, dstpath):
    """
    srcpath: result files before merge and nms
    dstpath: result files after merge and nms
    """
    # srcpath = r'E:\bod-dataset\results\bod-v3_rfcn_2000000'
    # dstpath = r'E:\bod-dataset\results\bod-v3_rfcn_2000000_nms'

    mergebase(srcpath,
              dstpath,
              py_cpu_nms)


def mergebypoly(srcpath, dstpath):
    """
    srcpath: result files before merge and nms
    dstpath: result files after merge and nms
    """
    # srcpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/comp4_test_results'
    # dstpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/testtime'

    # mergebase(srcpath,
    #           dstpath,
    #           py_cpu_nms_poly)
    
    # # 并行计算，多进程并行
    # mergebase_parallel(srcpath,
    #                    dstpath,
    #                    py_cpu_nms_poly_fast)
    # 不使用多进程
    mergebase(srcpath,
            dstpath,
            py_cpu_nms_poly_fast)

def secondNMS(srcpath, dstpath):
    """
    srcpath: result files before merge and nms
    dstpath: result files after merge and nms
    """
    # srcpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/comp4_test_results'
    # dstpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/testtime'

    filelist = util.GetFileFromThisRootDir(srcpath)
    # 是个字典，存储每个图像的检测结果，{img_name:[[x1...y4,conf, cls],...], ...}
    imgs_det_dict = {}
    for filename in filelist:
        # 类别的名称
        class_name = util.custombasename(filename)
        # print('name:', name)
        
        with open(filename, 'r') as f:
            
            lines = f.readlines()
            splitlines = [x.strip().split(' ') for x in lines]
            for splitline in splitlines:
                # 不包括后缀名
                img_name = splitline[0]

                confidence = float(splitline[1])
                points = list(map(float, splitline[2:10]))
                
                det = points + [confidence, class_name]
                if (img_name not in imgs_det_dict):
                    imgs_det_dict[img_name] = []
                imgs_det_dict[img_name].append(det)
        

    # 逐张图像进行NMS
    imgs_det_dict_after_NMS = {x: [] for x in imgs_det_dict}

    for img_name in imgs_det_dict:
        # 取出该图像中，所有的检测框，不包括目标类别
        boxes_points = [box[:-1] for box in imgs_det_dict[img_name]]
        keep = py_cpu_nms_poly_fast(np.array(boxes_points))
        # print('keep:', keep)
        outdets = []
        # print('nameboxdict[imgname]: ', nameboxnmsdict[imgname])
        for index in keep:
            # print('index:', index)
            outdets.append(imgs_det_dict[img_name][index])
        imgs_det_dict_after_NMS[img_name] = outdets
    
    ## 再把imgs_det_dict_after_NMS，按照目标类别的格式进行组织，以方便写入txt文件
    classes_det_dict = {}
    for img_name in imgs_det_dict_after_NMS:
        # boxes:[[x1...y4,conf, cls],...]
        boxes = imgs_det_dict_after_NMS[img_name]
        for box in boxes:
            class_name = box[-1]
            confidence = box[-2]

            box_out = img_name + ' ' + str(confidence) + ' ' + ' '.join(map(str, box[:-2])) + '\n'

            if class_name not in classes_det_dict:
                classes_det_dict[class_name] = []
            
            classes_det_dict[class_name].append(box_out)
    

    # 把NMS后的结果写入该类别文件
    for class_name in classes_det_dict:
        save_pathname = os.path.join(dstpath, class_name + '.txt')
        print(save_pathname)
        with open(save_pathname, 'w') as f:
            f.writelines(classes_det_dict[class_name])



if __name__ == '__main__':
    mergebyrec(r'work_dirs/temp/result_raw', r'work_dirs/temp/result_task2')
# mergebyrec()
