import numpy as np
import torch
from utils.general import norm_angle


# 根据offsets和anchor，解码得到旋转框
def delta2bbox_rotated_normalize(rois, deltas, means=(0., 0., 0., 0., 0.), stds=(1., 1., 1., 1., 1.), max_shape=None,
                       wh_ratio_clip=16 / 1000, clip_border=True):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.

    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 5)
        deltas (Tensor): Encoded offsets with respect to each roi.
            Has shape (N, 5 * num_classes). Note N = num_anchors * W * H when
            rois is a grid of anchors. Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates
        max_shape (tuple[int, int]): Maximum bounds for boxes. specifies (H, W)
        wh_ratio_clip (float): Maximum aspect ratio for boxes.
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.

    Returns:
        Tensor: Boxes with shape (N, 5), where columns represent

    References:
        .. [1] https://arxiv.org/abs/1311.2524
    """

    # 由于均值为0、方差为1，因此进行标准化前后没有任何变化
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means

    
    dx = denorm_deltas[:, 0::5]     # 从0开始，每隔5个取一个，取到最后，即取索引0、5、10......
    dy = denorm_deltas[:, 1::5]
    dw = denorm_deltas[:, 2::5]
    dh = denorm_deltas[:, 3::5]
    # 角度的offset
    dangle = denorm_deltas[:, 4::5]

    # 最大的长宽比
    max_ratio = np.abs(np.log(wh_ratio_clip))
    # clamp，进行截断
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    # expand_as，变成和dx相同的shape
    roi_x = (rois[:, 0]).unsqueeze(1).expand_as(dx)
    roi_y = (rois[:, 1]).unsqueeze(1).expand_as(dy)
    roi_w = (rois[:, 2]).unsqueeze(1).expand_as(dw)
    roi_h = (rois[:, 3]).unsqueeze(1).expand_as(dh)
    # anchor的角度
    roi_angle = (rois[:, 4]).unsqueeze(1).expand_as(dangle)
    
    # # 经过验证发现，输入参数rois和deltas所在的GPU就不一样，因此导致bug
    # print(f"rois:{rois.device}, deltas:{deltas.device}, dx:{dx.device}, dy:{dy.device}, roi_w:{roi_w.device}, roi_h:{roi_h.device}, roi_angle:{roi_angle.device}")
    # 这种x、y的编解码方法，从来没见过
    gx = dx * roi_w * torch.cos(roi_angle) \
         - dy * roi_h * torch.sin(roi_angle) + roi_x
    gy = dx * roi_w * torch.sin(roi_angle) \
         + dy * roi_h * torch.cos(roi_angle) + roi_y
    
    # 边界框宽高的编解码方法，与YOLOv3一样
    gw = roi_w * dw.exp()
    gh = roi_h * dh.exp()

    # 先获得解码后的角度，然后进行标准化，确保在[-0.25pi, 0.75pi)之间
    ga = np.pi * dangle + roi_angle
    ga = norm_angle(ga)

    bboxes = torch.stack([gx, gy, gw, gh, ga], dim=-1).view_as(deltas)
    return bboxes


# 根据offsets和anchor，解码得到旋转框
def delta2bbox_rotated(rois, deltas, is_encode_relative=True, wh_ratio_clip=16 / 1000):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.

    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 5)
        deltas (Tensor): Encoded offsets with respect to each roi.
            Has shape (N, 5). 
        wh_ratio_clip (float): Maximum aspect ratio for boxes.


    Returns:
        Tensor: Boxes with shape (N, 5), where columns represent

    References:
        .. [1] https://arxiv.org/abs/1311.2524
    """



    dx = deltas[:, 0]     # 从0开始，每隔5个取一个，取到最后，即取索引0、5、10......
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]
    # 角度的offset
    dangle = deltas[:, 4]

    # 最大的长宽比,np.log是以e为底的对数函数
    # wh_ratio_clip=16/1000，max_ratio为4.14
    # wh_ratio_clip=1e-6，wh_ratio_clip为13.2
    max_ratio = np.abs(np.log(wh_ratio_clip))

    # ##  查看截断的预测框和对应的anchors
    # inds_dw = (dw <= -max_ratio) | (dw>=max_ratio)
    # inds_dh = (dh <= -max_ratio) | (dh>=max_ratio)
    # inds = inds_dw | inds_dh
    # if torch.any(inds):
    #     print("截断")
    #     print("anchors:", rois[inds])
    #     print("截断前deltas:", deltas[inds])
    
    # clamp，进行截断
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)

    roi_x = rois[:, 0]
    roi_y = rois[:, 1]
    roi_w = rois[:, 2]
    roi_h = rois[:, 3]
    # anchor的角度
    roi_angle = rois[:, 4]
    
    # # 经过验证发现，输入参数rois和deltas所在的GPU就不一样，因此导致bug
    # print(f"rois:{rois.device}, deltas:{deltas.device}, dx:{dx.device}, dy:{dy.device}, roi_w:{roi_w.device}, roi_h:{roi_h.device}, roi_angle:{roi_angle.device}")
    # 这种x、y的编解码方法，从来没见过
    if is_encode_relative:
        cosa = torch.cos(roi_angle)
        sina = torch.sin(roi_angle)
        gx = dx * roi_w * cosa - dy * roi_h * sina + roi_x
        gy = dx * roi_w * sina + dy * roi_h * cosa + roi_y
    else:
        gx = dx * roi_w  + roi_x
        gy = dy * roi_h  + roi_y
    
    # 边界框宽高的编解码方法，与YOLOv3一样
    gw = roi_w * dw.exp()
    gh = roi_h * dh.exp()

    # 先获得解码后的角度，然后进行标准化，确保在[-0.25pi, 0.75pi)之间
    ga = np.pi * dangle + roi_angle
    ga = norm_angle(ga)

    bboxes = torch.stack([gx, gy, gw, gh, ga], dim=-1).view_as(deltas)

    # if torch.any(inds):
    #     print("解码后:", bboxes[inds])
    
    return bboxes


# 对旋转框进行编码
def rboxes_encode(anchors, gt_rboxes, is_encode_relative=True):
    '''Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            anchors (torch.Tensor): [N,5], x,y,w,h,theta, 单位为像素值, theta为弧度
            gt_bboxes (torch.Tensor): [N,5], x,y,w,h,theta, 单位为像素值, theta为弧度

        Returns:
            encoded_rboxes(torch.Tensor): Box transformation deltas
        
    '''
    # 保证anchor的个数个gt的个数一样
    assert anchors.size(0) == gt_rboxes.size(0)
    assert anchors.size(-1) == gt_rboxes.size(-1) == 5

    # 下面进行编码
    gt_w = gt_rboxes[..., 2]
    gt_h = gt_rboxes[..., 3]
    gt_angle = gt_rboxes[..., 4]

    anchors_w = anchors[..., 2]
    anchors_h = anchors[..., 3]
    anchors_angles = anchors[..., 4]

    # gt与anchor的中心点坐标的offsets
    xy_offsets = gt_rboxes[..., 0:2] - anchors[..., 0:2]

    ## 计算offsets
    # 由于是方形anchor，且角度为0，所以anchors_w确实是水平方向的长度，可以用来标准化，但是无法确保归一化
    # 此外，这里的xy的offsets的计算不太理解，不知道为什么角度参与计算；
    # 杨学的R3det算法等都不是采用的这种方法，而ROITransformer等就采用的类似的方法
    if is_encode_relative:
        cosa = torch.cos(anchors_angles)
        sina = torch.sin(anchors_angles)
        dx = (cosa * xy_offsets[..., 0] + sina * xy_offsets[..., 1]) / anchors_w
        dy = (-sina * xy_offsets[..., 0] + cosa * xy_offsets[..., 1]) / anchors_h
    else:
        dx = xy_offsets[..., 0] / anchors_w
        dy = xy_offsets[..., 1] / anchors_h
    # torch.log是以e为底的，标准化，但是无法确保归一化
    dw = torch.log(gt_w / anchors_w)
    dh = torch.log(gt_h / anchors_h)
    
    # 角度差转化到[-0.25pi,0.75pi]之间，然后进行归一化
    da = (gt_angle - anchors_angles)
    da = norm_angle(da) / np.pi

    encoded_rboxes = torch.stack((dx, dy, dw, dh, da), -1).reshape(-1,5)

    # means = deltas.new_tensor(means).unsqueeze(0)
    # stds = deltas.new_tensor(stds).unsqueeze(0)
    # deltas = deltas.sub_(means).div_(stds)

    return encoded_rboxes


def rboxes_decode(anchors,
               pred_bboxes,
               is_encode_relative=True,
               wh_ratio_clip=16 / 1000 ):
    """Apply transformation `pred_bboxes` to `boxes`.

    Args:
        boxes (torch.Tensor): Basic boxes.
        pred_bboxes (torch.Tensor): Encoded boxes with shape
        max_shape (tuple[int], optional): Maximum shape of boxes.
            Defaults to None.
        wh_ratio_clip (float, optional): The allowed ratio between
            width and height.

    Returns:
        torch.Tensor: Decoded boxes.
    """
 
    assert pred_bboxes.size(0) == anchors.size(0)
    assert anchors.size(-1) == pred_bboxes.size(-1) == 5

    decoded_rboxes = delta2bbox_rotated(anchors, pred_bboxes, is_encode_relative=is_encode_relative, wh_ratio_clip=wh_ratio_clip)

    return decoded_rboxes

