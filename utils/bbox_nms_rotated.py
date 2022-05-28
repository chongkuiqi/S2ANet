import torch

from .ml_nms_rotated import ml_nms_rotated


def multiclass_nms_rotated(bboxes,
                           scores,
                           score_thr=0.05,
                           iou_thr=0.5,
                           max_per_img = 2000):
    """NMS for multi-class bboxes.
    Args:
        multi_bboxes (Tensor): shape (n, 5)
        multi_scores (Tensor): shape (n, num_classes), don't have the background class
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        iou_thr (float): NMS IoU threshold
        max_per_img (int): if there are more than max_per_img bboxes after NMS,
            only top max_per_img will be kept.

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 6) and (k, 1). Labels
            are 0-based.
    """
    num_classes = scores.size(1)

    # 增加边界框的个数，即一个特征位置点上，共有num_classes个框，代表不同类别的框
    assert bboxes.shape[1] == 5
    # bboxes shape:[N, num_classes, 5]
    bboxes = bboxes[:, None].expand(-1, num_classes, 5)

    ## 根据类别分数进行阈值过滤
    # score shape [N,num_classes]
    mask = scores > score_thr
    # scores shape [N2]
    scores = scores[mask]
    # bboxes shape [N2, 5]
    bboxes = bboxes[mask]

    # .nonzero()返回非零位置的横纵坐标
    labels = mask.nonzero(as_tuple=False)[:, 1]
    # 放在同一个设备上
    labels = labels.to(bboxes)

    
    if bboxes.shape[0] > 0:
        # 进行多类别的NMS
        keep = ml_nms_rotated(bboxes, scores, labels, iou_thr)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        if keep.size(0) > max_per_img:
            _, inds = scores.sort(descending=True)
            inds = inds[:max_per_img]
            bboxes = bboxes[inds]
            scores = scores[inds]
            labels = labels[inds]
        return torch.cat([bboxes, scores[:, None]], dim=1), labels
    # 如果进行上述处理后，没有目标
    else:
        # boxes shape为[N,6], x,y,w,h,theta,score
        bboxes = bboxes.new_zeros((0, 6))
        labels = bboxes.new_zeros((0,1), dtype=torch.long)
        return bboxes, labels




# def multiclass_nms_rotated(multi_bboxes,
#                            multi_scores,
#                            score_thr=0.05,
#                            iou_thr=0.5,
#                            max_before_nms = 2000,
#                            max_per_img = 2000,
#                            score_factors=None):
#     """NMS for multi-class bboxes.
#     Args:
#         multi_bboxes (Tensor): shape (n, 5)
#         multi_scores (Tensor): shape (n, background+num_classes), where the 0th column
#             contains scores of the background class, but this will be ignored.
#         score_thr (float): bbox threshold, bboxes with scores lower than it
#             will not be considered.
#         iou_thr (float): NMS IoU threshold
#         max_before_nms (int): if there are more than max_before_nms bboxes before NMS,
#             only top max_before_nms will be kept.
#         max_per_img (int): if there are more than max_per_img bboxes after NMS,
#             only top max_per_img will be kept.
#         score_factors (Tensor): The factors multiplied to scores before
#             applying NMS
#     Returns:
#         tuple: (bboxes, labels), tensors of shape (k, 6) and (k, 1). Labels
#             are 0-based.
#     """
#     num_classes = multi_scores.size(1) - 1
#     # exclude background category，排除背景分数
#     scores = multi_scores[:, 1:].reshape(-1, num_classes)
    
#     ## 考虑到第三方库实现的NMS，可能要求输入的boxes和score都必须是排好序的，
#     # 因此我们保持先排序挑选出分数最大topk个框、再阈值过滤的处理顺序不变，最后进行NMS
#     # 先排序挑选出分数最大topk个框
#     max_scores, _ = scores.max(dim=1)
#     _, topk_inds = max_scores.topk(max_before_nms)
#     # 取出候选的框、分类分数
#     scores = scores[topk_inds, :] # shape:[N, num_classes]
#     bboxes = multi_bboxes[topk_inds, :]     # shape:[N, 5]


#     # 再进行阈值过滤
#     max_scores, max_scores_inds = scores.max(dim=1)
#     # 根据分数阈值进行过滤
#     # filter out boxes with low scores
#     score_mask = max_scores > score_thr
#     if score_factors is not None:
#         scores = scores * score_factors[:, None]
#     # shape ：[N2]
#     scores = max_scores[score_mask]
#     # shape:[N2,5]
#     bboxes = bboxes[score_mask]
#     # 类别id
#     labels = max_scores_inds[score_mask]

    
#     if bboxes.shape[0] > 0:
#         # 进行多类别的NMS
#         keep = ml_nms_rotated(bboxes, scores, labels, iou_thr)
#         bboxes = bboxes[keep]
#         scores = scores[keep]
#         labels = labels[keep]

#         if keep.size(0) > max_per_img:
#             _, inds = scores.sort(descending=True)
#             inds = inds[:max_per_img]
#             bboxes = bboxes[inds]
#             scores = scores[inds]
#             labels = labels[inds]
#         return torch.cat([bboxes, scores[:, None]], dim=1), labels
#     # 如果进行上述处理后，没有目标
#     else:
#         # boxes shape为[N,6], x,y,w,h,theta,score
#         bboxes = multi_bboxes.new_zeros((0, 6))
#         labels = multi_bboxes.new_zeros((0,1), dtype=torch.long)
#         return bboxes, labels

