import torch
from utils.metrics import bbox_iou_rotated
from functools import partial


def multi_apply(func, *args, **kwargs):
    # 将函数func的参数kwargs固定，返回新的函数pfunc
    pfunc = partial(func, **kwargs) if kwargs else func
    # 这里的args表示feats和anchor_strides两个序列，map函数会分别遍历这两个序列，然后送入pfunc函数
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def split_to_levels(assign_gt_ids, num_anchors_every_level):
    '''
      将所有特征层级的正负样本分配结果，拆分为一个列表，存储各个特征层级的分配结果

    assign_gt_ids : shape:(all_levels_anchors)
    num_anchors_every_level: 每一个特征层级的网格anchors的个数
    '''

    level_assign_gt_ids = []
    start = 0
    for n in num_anchors_every_level:
        end = start + n
        level_assign_gt_ids.append(assign_gt_ids[start:end].squeeze(0))
        start = end
    
    # 是个列表，存储每一个特征层级的正负样本分配结果
    return level_assign_gt_ids


def assign_labels(anchors, gt_boxes, imgs_size=(1024,1024), 
        pos_iou_thr=0.5, neg_iou_thr=0.4, min_pos_iou_thr=0, gt_max_assign_all=True,
        filter_invalid_anchors=True, filter_invalid_ious=True
    ):
    '''
    anchors shape : [M, 5], x y w h 都是以像素为单位的，角度为弧度，并且不是归一化值
    gt_boxes shape : [N, 5]

    min_pos_iou_thr: 对于那些与所有anchor的IOU都小于正样本阈值的gt，分配最大IOU的anchor来预测这个gt，
                    但是这个最大IOU也不能太小，于是设置这个阈值
    gt_max_assign_all: 上述情况下，gt可能与多个anchor都具有最大IOU，此时就把多个anchor都设置为正样本
    filter_invalid_anchors: invalid_anchor是指参数超出图像边界的anchor，无效anchors设置为忽略样本
    filter_invalid_ious ：旋转iou的计算函数有问题，范围不在[0,1.0]之间，将超出范围的iou设置为负值，从而成为忽略样本
    return:
        assign_gt_ids : [M], -2表示是忽略样本，-1表示是负样本，大于等于0表示是正样本，具体的数值表示是预测的哪个gt
    '''


    # 必须保证传入bbox_iou_rotated的参数的内存连续性
    # 否则不会报错，但会出现计算错误，这种bug更难排查
    if not anchors.is_contiguous():
        anchors = anchors.contiguous()
    if not gt_boxes.is_contiguous():
        gt_boxes = gt_boxes.contiguous()

    device = gt_boxes.device

    num_anchors = anchors.shape[0]
    num_gt_boxes = gt_boxes.shape[0]

    # 标签分配的结果，一维向量，每个位置表示一个anchor，其值表示该anchor预测哪一个gt，默认全是忽略样本
    assign_gt_ids = torch.ones(num_anchors, dtype=torch.long, device=device)*(-2)

    # ## 判断anchor是否有超出图像边界的情况
    # # 对于init_anchors没有这种问题，但对于refine_anchors可能存在超出边界的问题
    if filter_invalid_anchors:
        flags = (anchors[:, 0] >= 0) & \
                (anchors[:, 1] >= 0) & \
                (anchors[:, 0] <= imgs_size[1]) & \
                (anchors[:, 1] <= imgs_size[0]) & \
                (anchors[:, 2] < imgs_size[1]) & \
                (anchors[:, 3] < imgs_size[0])    
    
    # 首先判断gt_boxes是否为空
    if num_gt_boxes == 0:
        # 如果为空，那么anchors就只有可能是负样本，或者因为anchors无效而成为忽略样本
        # 将有效的anchors设置为负样本
        if filter_invalid_anchors:
            assign_gt_ids[flags] = -1
        else:
            assign_gt_ids[:] = -1

        return assign_gt_ids

    ious = bbox_iou_rotated(anchors, gt_boxes)

    if filter_invalid_ious:
        # assert torch.all((ious>=0) & (ious<=1))
        # "iou 的值应该大于等于0，并且小于等于1"
        if not torch.all((ious>=0) & (ious<=1)):
            inds = (ious<0) | (ious>1)
            # print(ious[inds])
            # # 设置为负值, 则可以保证这一对iou值不会被标注为正样本或负样本
            ious[inds] = -0.5

        #     # print(f"处于{mode}模块下，有几个iou计算不准确")   
    
    # 无效的anchors的iou设置为负值，使该anchors成为忽略样本
    if filter_invalid_anchors:
        ious[~flags] = -0.5

    # # 打印一下iou比较大的anchors和gt
    # if torch.any(ious > 0.98):
    #     inds_y, inds_x = torch.where(ious > 0.98)

    #     print("iou>0.98")
    #     print("anchors:", anchors[inds_y])
    #     print("gt:", gt_boxes[inds_x])

    # 下面对各种可能存在的情况进行处理
    # 1.首先分配负样本，负样本的定义，与所有的gt的IOU都小于负样本阈值的anchor，也就是最大值都小于负样本阈值
    # 如果最大值有2个，.max()函数只会返回第一个最大值的坐标索引
    max_ious, argmax_ious = ious.max(dim=1)
    assign_gt_ids[(max_ious>=0)&(max_ious<neg_iou_thr)] = -1
    


    # 2.然后分配正样本，正样本有两种定义
    # （1）正样本1：一个anchor与某些gt的IOU大于pos_iou_thr：可能与多个gt的IOU都大于阈值，只选择最大IOU的gt分配给该anchor
    # 如果最大IOU值有2个，.max()函数只会返回第一个最大值的坐标索引，因此这里我们会忽略其它的gt框
    pos_inds = max_ious >= pos_iou_thr
    assign_gt_ids[pos_inds] = argmax_ious[pos_inds]


    # （2）正样本2：一个gt与所有anchor的IOU都小于正样本阈值，但是又不能不预测这个gt，所以把他分配给具有最大IOU的anchor，这个anchor也是正样本
    # 有可能具有最大IOU阈值的anchor有多个，那么这些anchor都可以作为正样本
    # 如果最大值有2个，.max()函数只会返回第一个最大值的坐标索引
    gt_max_ious, gt_argmax_ious = ious.max(dim=0)
    # 逐个gt框进行处理
    for i in range(num_gt_boxes):
        # 我认为这个地方设置为大于更合适，因为如果是>=，且min_pos_iou_thr=0.0，那么可能iou=0的也设置为正样本了
        if gt_max_ious[i] > min_pos_iou_thr:
            if gt_max_assign_all:
                # 有可能具有最大IOU阈值的anchor有多个，那么这些anchor都可以作为正样本
                # 找到具有最大iou的anchor
                max_iou_ids = ious[:,i] == gt_max_ious[i]
                # max_iou_ids = (ious[:,i] > 0) & (ious[:,i] > gt_max_ious[i] - 1e-2)
                # 将该gt分配给这些anchor
                assign_gt_ids[max_iou_ids] = i

            else:
                assign_gt_ids[gt_argmax_ious[i]] = i

    
    return assign_gt_ids

