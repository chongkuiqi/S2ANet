# --------------------------------------------------------
# dota_evaluation_task1
# Licensed under The MIT License [see LICENSE for details]
# Written by Jian Ding, based on code from Bharath Hariharan
# --------------------------------------------------------

"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 15 result files, for the format, you can refer to "http://captain.whu.edu.cn/DOTAweb/tasks.html"
    search for PATH_TO_BE_CONFIGURED to config the paths
    Note, the evaluation is on the large scale images
"""
# import cPickle
import numpy as np

try:
    from polyiou import polyiou
except:
    from DOTA_devkit.polyiou import polyiou


def parse_gt(filename):
    """
    :param filename: ground truth file to parse
    :return: all instances in a picture
    """
    # objects是个列表，每个列表元素都是个字典，每个字典结果为{'name':class_name(str), 'difficult':0 or 1(int), 'bbox':(list)[8个坐标，float]}
    objects = []
    with  open(filename, 'r') as f:
        while True:
            # 读取一行
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                object_struct = {}
                if (len(splitlines) < 9):
                    continue
                object_struct['name'] = splitlines[8]

                if (len(splitlines) == 9):
                    object_struct['difficult'] = 0
                elif (len(splitlines) == 10):
                    object_struct['difficult'] = int(splitlines[9])
                object_struct['bbox'] = [float(splitlines[0]),
                                         float(splitlines[1]),
                                         float(splitlines[2]),
                                         float(splitlines[3]),
                                         float(splitlines[4]),
                                         float(splitlines[5]),
                                         float(splitlines[6]),
                                         float(splitlines[7])]
                objects.append(object_struct)
            else:
                break
    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             is_filter_difficult=True,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    # if not os.path.isdir(cachedir):
    #   os.mkdir(cachedir)
    # cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images

    # imagesetfile是个.txt文件，存储测试集图像名字，不包括拓展名
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    # 图像名字列表（不包括拓展名）
    imagenames = [x.strip() for x in lines]

    # 读取每张图像的gt标签
    recs = {}
    for i, imagename in enumerate(imagenames):
        # print('parse_files name: ', annopath.format(imagename))
        # parse_gt返回个列表，每个列表元素都是个字典，每个字典结构为：
        # {'name':class_name(str), 'difficult':0 or 1(int), 'bbox':(list)[8个坐标，float]}
        recs[imagename] = parse_gt(annopath.format(imagename))

    # 找到整个数据集中，仅仅属于当前目标类别的标签
    # extract gt objects for this class
    class_recs = {}
    num_gts = 0
    for imagename in imagenames:
        # R是当前图像的当前类别的标签，不是所有目标类别的标签
        # classname就是当前要统计的类别，也就是只把当前类别的gt提取出来
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        # 存放检测状态，意思是，如果该gt box已经分配了检测box，就置为True
        det = [False] * len(R)
        
        if is_filter_difficult:
            # 注意，是取反相加，表示的是简单样本的个数
            num_gts = num_gts + sum(~difficult)
        else:
            num_gts = num_gts + difficult.shape[0]
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # 获取类别该classname的所有检测结果
    # read dets from Task1* files
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()


    # 检测框的个数
    num_dets = len(lines)

    # 如果存在检测框
    if num_dets:
        
        splitlines = [x.strip().split(' ') for x in lines]
        # 该检测框所在的图像
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        # 获得8个坐标
        pred_bboxes = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        # np.argsort()是从小到大排序，这里对置信度取反再排序，意思就是对置信度从大到小排序
        
        sorted_ind = np.argsort(-confidence)
        # print(confidence.shape)
        # print(sorted_ind)
        # print(classname)


        # sorted_scores = np.sort(-confidence)
        sorted_scores = confidence[sorted_ind]
        
        ## note the usage only in numpy not for list
        # 根据置信度对box从大到小排序
        pred_bboxes = pred_bboxes[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
        
        # go down dets and mark TPs and FPs
        # 标记哪些是TP，哪些是FP
        tp = np.zeros(num_dets)
        fp = np.zeros(num_dets)
        for pred_box_idx in range(num_dets):
            # 取出该pred box所在图像的所有的、属于该类别的gt boxes，
            # R是个字典，{'bbox': bbox,'difficult': difficult,'det': det}
            R = class_recs[image_ids[pred_box_idx]]
            # 取出该pred box
            pred_bbox = pred_bboxes[pred_box_idx, :].astype(float)
            ovmax = -np.inf
            # 取出所有的gt boxes
            gt_bboxes = R['bbox'].astype(float)

            ## compute det bb with each BBGT
            if gt_bboxes.size > 0:
                # compute overlaps
                # intersection

                # 先计算水平框之间的IOU，如果水平框之间IOU都是0，那么旋转框之间IOU就肯定是0
                # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
                # pdb.set_trace()
                gt_boxes_xmin = np.min(gt_bboxes[:, 0::2], axis=1)
                gt_boxes_ymin = np.min(gt_bboxes[:, 1::2], axis=1)
                gt_boxes_xmax = np.max(gt_bboxes[:, 0::2], axis=1)
                gt_boxes_ymax = np.max(gt_bboxes[:, 1::2], axis=1)
                pred_box_xmin = np.min(pred_bbox[0::2])
                pred_box_ymin = np.min(pred_bbox[1::2])
                pred_box_xmax = np.max(pred_bbox[0::2])
                pred_box_ymax = np.max(pred_bbox[1::2])

                ixmin = np.maximum(gt_boxes_xmin, pred_box_xmin)
                iymin = np.maximum(gt_boxes_ymin, pred_box_ymin)
                ixmax = np.minimum(gt_boxes_xmax, pred_box_xmax)
                iymax = np.minimum(gt_boxes_ymax, pred_box_ymax)
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                # 交集部分的面积
                inters = iw * ih

                # 并集部分的面积
                # union
                uni = ((pred_box_xmax - pred_box_xmin + 1.) * (pred_box_ymax - pred_box_ymin + 1.) +
                    (gt_boxes_xmax - gt_boxes_xmin + 1.) *
                    (gt_boxes_ymax - gt_boxes_ymin + 1.) - inters)

                overlaps = inters / uni

                # 只取出和当前pred box水平框IOU大于0的gt boxes
                gt_boxes_keep_mask = overlaps > 0
                gt_boxes_keep = gt_bboxes[gt_boxes_keep_mask, :]
                gt_boxes_keep_index = np.where(overlaps > 0)[0]

                def calcoverlaps(BBGT_keep, bb):
                    overlaps = []
                    for index, GT in enumerate(BBGT_keep):
                        overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                        overlaps.append(overlap)
                    return overlaps

                if len(gt_boxes_keep) > 0:
                    # 计算旋转框的IOU
                    overlaps = calcoverlaps(gt_boxes_keep, pred_bbox)
                    # 最大的旋转IOU值
                    ovmax = np.max(overlaps)
                    max_gt_box_idx = np.argmax(overlaps)
                    # pdb.set_trace()
                    max_gt_box_idx = gt_boxes_keep_index[max_gt_box_idx]

            # 如果最大旋转IOU值，大于设定的IOU阈值，说明该pred box就分配到了这个gt box，这个pred_box是TP，否则是FP
            if ovmax > ovthresh:
                # 如果不让困难样本参与计算mAP
                if is_filter_difficult:
                    # 判断是否是困难样本，如果是，就忽略，既不是正样本，也不是负样本
                    if not R['difficult'][max_gt_box_idx]:
                        # 判断该gt box是否已经分配了pred box，
                        # 如果否，就把pred box作为TP；否则，置为FP
                        if not R['det'][max_gt_box_idx]:
                            tp[pred_box_idx] = 1.
                            R['det'][max_gt_box_idx] = 1
                        else:
                            fp[pred_box_idx] = 1.
                # 困难样本参与计算mAP
                else:
                    # 判断该gt box是否已经分配了pred box，
                    # 如果否，就把pred box作为TP；否则，置为FP
                    if not R['det'][max_gt_box_idx]:
                        tp[pred_box_idx] = 1.
                        R['det'][max_gt_box_idx] = 1
                    else:
                        fp[pred_box_idx] = 1.
            else:
                fp[pred_box_idx] = 1.
        
        # compute precision recall
        # print('check fp:', fp)
        # print('check tp', tp)

        # print('npos num:', npos)
        # FP、TP的累积分布
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        # recall曲线
        rec = tp / float(num_gts)
        # precision曲线
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)

        

    # 如果不存在检测框
    else:
        # 如果整个数据集中该类别的gt框的个数也等于0,这种情况下，应该没办法计算P和R
        if num_gts == 0:
            print(f"类别{classname} 的检测框个数为0，gt框个数也为0，不应该出现这种情况")
            exit()
        else:
            # 召回率、准确率、ap值均为0
            rec, prec, ap, sorted_scores = np.zeros(1), np.zeros(1), np.zeros(1).sum(), np.zeros(1)

    return rec, prec, ap, sorted_scores


def main():
    detpath = r'/home/hjm/mmdetection/work_dirs/cascade_s2anet_r50_fpn_1x_dota/results_after_nms/{:s}.txt'
    annopath = r'data/dota/test/labelTxt/{:s}.txt'  # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
    imagesetfile = r'data/dota/test/test.txt'

    # For DOTA-v1.5
    # classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    #             'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']
    # For DOTA-v1.0
    classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
                  'tennis-court',
                  'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool',
                  'helicopter']
    classaps = []
    map = 0
    for classname in classnames:
        print('classname:', classname)
        rec, prec, ap = voc_eval(detpath,
                                 annopath,
                                 imagesetfile,
                                 classname,
                                 ovthresh=0.5,
                                 use_07_metric=True)
        map = map + ap
        # print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
        print('ap: ', ap)
        classaps.append(ap)

        # # umcomment to show p-r curve of each category
        # plt.figure(figsize=(8,4))
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.xticks(fontsize=11)
        # plt.yticks(fontsize=11)
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        # ax = plt.gca()
        # ax.spines['top'].set_color('none')
        # ax.spines['right'].set_color('none')
        # plt.plot(rec, prec)
        # # plt.show()
        # plt.savefig('pr_curve/{}.png'.format(classname))
    map = map / len(classnames)
    print('map:', map)
    classaps = 100 * np.array(classaps)
    print('classaps: ', classaps)


if __name__ == '__main__':
    main()
