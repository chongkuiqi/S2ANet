# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel

from copy import deepcopy

from utils.metrics import bbox_iou_rotated

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch




class SmoothL1Loss(nn.Module):

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        # 这里的reduction，指的是所有的旋转框的总损失进行平均,而不是一个旋转框产生的x/y/w/h/theta/进行平均
        self.reduction = reduction
        self.loss_weight = loss_weight

        assert self.reduction in (None, 'none', 'mean', 'sum')
    
    def smooth_l1_loss(self, pred, target):
        assert self.beta > 0
        assert pred.size() == target.size() and target.numel() > 0
        diff = torch.abs(pred - target)
        
        # torch.where(condition, x, y)，满足条件返回x，不满足返回y
        loss = torch.where(diff < self.beta, 0.5 * diff * diff / self.beta,
                        diff - 0.5 * self.beta)

        # 一个旋转框的x/y/w/h/theta/损失加和，作为一个旋转框的回归损失
        loss = loss.sum(dim=1)
        return loss

    def forward(self, pred, target):
        '''
        pred shape:[N,5],每一行是一个旋转框
        target shape: [N,5]
        '''
        
        # loss shape : [N]
        loss = self.loss_weight * self.smooth_l1_loss(pred, target)
        
        # exit()
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
        



# mmdetection框架版本的focal loss
from .sigmoid_focal_loss import sigmoid_focal_loss

class mmFocalLoss(nn.Module):
    
    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(mmFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

        assert self.reduction in (None, 'none', 'mean', 'sum')

    def forward(self, pred, targets):
        
        ## # C++程序中要求target是长整型
        targets = targets.float().long()
        # 0-index，类别id从0开始
        
        max_value, gt_id = targets.max(dim=1)
        # 1-index，类别id从1开始
        gt_id[max_value>0] = gt_id[max_value>0] + 1


        if self.use_sigmoid:
            
            loss = self.loss_weight * sigmoid_focal_loss(pred, gt_id, self.gamma, self.alpha)
            
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:  # 'none'
                return loss
            
        else:
            raise NotImplementedError




################ yangxue的KLD损失

def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    # wh = xywhr[..., 2:4].clamp(min=1e-5, max=5e4).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                            1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma


def gwd_loss(pred, target, fun='sqrt', tau=2.0):
    """Gaussian Wasserstein distance loss.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """
    mu_p, sigma_p = pred
    mu_t, sigma_t = target

    xy_distance = (mu_p - mu_t).square().sum(dim=-1)

    whr_distance = sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_distance = whr_distance + sigma_t.diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    _t_tr = (sigma_p.bmm(sigma_t)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = (sigma_p.det() * sigma_t.det()).clamp(0).sqrt()
    whr_distance += (-2) * (_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt()

    dis = xy_distance + whr_distance
    gwd_dis = dis.clamp(min=1e-6)

    if fun == 'sqrt':
        loss = 1 - 1 / (tau + torch.sqrt(gwd_dis))
    elif fun == 'log1p':
        loss = 1 - 1 / (tau + torch.log1p(gwd_dis))
    else:
        scale = 2 * (_t_det_sqrt.sqrt().sqrt()).clamp(1e-7)
        loss = torch.log1p(torch.sqrt(gwd_dis) / scale)
    return loss



def bcd_loss(pred, target, fun='log1p', tau=1.0):
    """Bhatacharyya distance loss.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """
    mu_p, sigma_p = pred
    mu_t, sigma_t = target

    mu_p = mu_p.reshape(-1, 2)
    mu_t = mu_t.reshape(-1, 2)
    sigma_p = sigma_p.reshape(-1, 2, 2)
    sigma_t = sigma_t.reshape(-1, 2, 2)

    delta = (mu_p - mu_t).unsqueeze(-1)
    sigma = 0.5 * (sigma_p + sigma_t)
    sigma_inv = torch.inverse(sigma)

    term1 = torch.log(
        torch.det(sigma) /
        (torch.sqrt(torch.det(sigma_t.matmul(sigma_p))))).reshape(-1, 1)
    term2 = delta.transpose(-1, -2).matmul(sigma_inv).matmul(delta).squeeze(-1)
    dis = 0.5 * term1 + 0.125 * term2
    bcd_dis = dis.clamp(min=1e-6)

    if fun == 'sqrt':
        loss = 1 - 1 / (tau + torch.sqrt(bcd_dis))
    elif fun == 'log1p':
        loss = 1 - 1 / (tau + torch.log1p(bcd_dis))
    else:
        loss = 1 - 1 / (tau + bcd_dis)
    return loss


def kld_loss(pred, target, fun='log1p', tau=1.0):
    """Kullback-Leibler Divergence loss.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    # mu_p shape : (N,2), sigma_p shape : (N, 2, 2)
    mu_p, sigma_p = pred
    mu_t, sigma_t = target

    mu_p = mu_p.reshape(-1, 2)
    mu_t = mu_t.reshape(-1, 2)
    sigma_p = sigma_p.reshape(-1, 2, 2)
    sigma_t = sigma_t.reshape(-1, 2, 2)

    # unsqueeze增加一个维度，-1表示在倒数第一个维度上增加维度，因此delta的shape为(N,2,1)
    delta = (mu_p - mu_t).unsqueeze(-1)

    # inverse为求逆矩阵
    sigma_t_inv = torch.inverse(sigma_t)
    # matmul是矩阵乘法，transpose为矩阵转置，倒数第一维度和倒数第二维度进行转置
    # 
    term1 = delta.transpose(-1,
                            -2).matmul(sigma_t_inv).matmul(delta).squeeze(-1)
    term2 = torch.diagonal(
        sigma_t_inv.matmul(sigma_p),
        dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) + \
        torch.log(torch.det(sigma_t) / torch.det(sigma_p)).reshape(-1, 1)
    dis = term1 + term2 - 2
    # 截断，防止其为负数
    kl_dis = dis.clamp(min=1e-6)

    if fun == 'sqrt':
        kl_loss = 1 - 1 / (tau + torch.sqrt(kl_dis))
    else:
        kl_loss = 1 - 1 / (tau + torch.log1p(kl_dis))
    return kl_loss


class GDLoss_v1(nn.Module):
    """Gaussian based loss.

    Args:
        loss_type (str):  Type of loss.
        fun (str, optional): The function applied to distance.
            Defaults to 'log1p'.
        tau (float, optional): Defaults to 1.0.
        reduction (str, optional): The reduction method of the
            loss. Defaults to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """
    BAG_GD_LOSS = {'kld': kld_loss, 'bcd': bcd_loss, 'gwd': gwd_loss}

    def __init__(self,
                 loss_type,
                 fun='sqrt',
                 tau=1.0,
                 reduction='mean',
                 loss_weight=1.0,
                 ):
        super(GDLoss_v1, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        assert fun in ['log1p', 'sqrt', '']
        assert loss_type in self.BAG_GD_LOSS


        self.loss = self.BAG_GD_LOSS[loss_type]
        
        # 将旋转框转化为二维高斯分布
        self.preprocess = xy_wh_r_2_xy_sigma
        self.fun = fun
        self.tau = tau


        self.reduction = reduction
        self.loss_weight = loss_weight

        assert self.reduction in (None, 'none', 'mean', 'sum')

    def forward(self,
                pred,
                target,
                ):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted convexes.
            target (torch.Tensor): Corresponding gt convexes.
        """
        
        # 旋转框转化为二维高斯分布
        pred = self.preprocess(pred)
        target = self.preprocess(target)

        # loss shape : [N,1]
        loss = self.loss(pred, target, fun=self.fun, tau=self.tau) * self.loss_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


from .box_iou_rotated_diff import box_iou_rotated_differentiable
class RotatedIoULoss(nn.Module):

    def __init__(self, linear=False, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(RotatedIoULoss, self).__init__()
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        
        assert self.reduction in (None, 'none', 'mean', 'sum')

    def iou_loss(self, pred, target):
        """IoU loss.

        Computing the IoU loss between a set of predicted bboxes and target bboxes.
        The loss is calculated as negative log of IoU.

        Args:
            pred (Tensor): Predicted bboxes of format (x, y, w, h, a),
                shape (n, 5).
            target (Tensor): Corresponding gt bboxes, shape (n, 5).
            linear (bool):  If True, use linear scale of loss instead of
                log scale. Default: False.
            eps (float): Eps to avoid log(0).

        Return:
            Tensor: Loss tensor.
        """
        ious = box_iou_rotated_differentiable(pred, target).clamp(min=self.eps)
        
        # print("pred:", pred.shape)
        # print("target:", target.shape)
        # print("ious:", ious)

        if self.linear:
            loss = 1 - ious
        else:
            loss = -ious.log()
        return loss

    def forward(self, pred,target):

        
        loss = self.loss_weight * self.iou_loss(pred, target)

        # exit()
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
        

class LARLoss(nn.Module):
    
    def __init__(self, ar_range=[3.0,10.0], reduction='mean', loss_weight=1.0):
        super(LARLoss, self).__init__()
        # 这里的reduction，指的是所有的旋转框的总损失进行平均,而不是一个旋转框产生的x/y/w/h/theta/进行平均
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.ar_range = ar_range

        assert self.reduction in (None, 'none', 'mean', 'sum')
    
    def lar_loss(self, pred):
        
        # pred[:,2] > 3*pred[:,3]
        ar = torch.abs(pred[:,2] / pred[:,3])

        zero = torch.zeros(1,device=ar.device)

        # torch.where(condition, x, y)，满足条件返回x，不满足返回y
        loss = torch.where(ar < self.ar_range[0], self.ar_range[0]-ar, zero) + torch.where(ar > self.ar_range[1], ar-self.ar_range[0], zero)

        # # 一个旋转框的x/y/w/h/theta/损失加和，作为一个旋转框的回归损失
        # loss = loss.sum(dim=1)
        return loss

    def forward(self, pred):
        '''
        pred shape:[N,5],每一行是一个旋转框
        '''
        
        # loss shape : [N]
        loss = self.loss_weight * self.lar_loss(pred)
        
        # exit()
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
   