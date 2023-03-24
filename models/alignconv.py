import torch
import torch.nn as nn

from .dcn import DeformConv
from models.init_weights import normal_init
from torch.nn.modules.utils import _pair

class AlignConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=1):
        super(AlignConv, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.padding = tuple((size - 1)//2 for size in self.kernel_size)
        self.deform_conv = DeformConv(in_channels,
                                      out_channels,
                                      kernel_size=self.kernel_size,
                                      padding=self.padding,
                                      deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        normal_init(self.deform_conv, std=0.01)

    # 根据anchors，找采样点位置的偏移
    @torch.no_grad()
    def get_offset(self, anchors, featmap_size, stride):
        # anchors shape [H*W, 5]
        dtype, device = anchors.dtype, anchors.device
        feat_h, feat_w = featmap_size
        
        # kernel_size=3, pad=1        
        pady = (self.kernel_size[0] - 1) // 2
        idy = torch.arange(-pady, pady + 1, dtype=dtype, device=device)
        padx = (self.kernel_size[1] - 1) // 2
        # idx : tensor([-1,  0,  1])，标准卷积9个采样点位置的偏移
        idx = torch.arange(-padx, padx + 1, dtype=dtype, device=device)

        # 根据idx构建网格，yy/xx shape [3,3]
        yy, xx = torch.meshgrid(idy, idx)
        # shape [3*3]
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)

        # 标准卷积的采样点位置
        # get sampling locations of default conv
        xc = torch.arange(0, feat_w, device=device, dtype=dtype)
        yc = torch.arange(0, feat_h, device=device, dtype=dtype)
        yc, xc = torch.meshgrid(yc, xc)
        # shape [feat_h*feat_w]
        xc = xc.reshape(-1)
        yc = yc.reshape(-1)
        # 标准卷积在当前特征点位置进行采样，每个特征点位置有9个采样点
        # shape [feat_h*feat_w, 3*3]
        x_conv = xc[:, None] + xx
        y_conv = yc[:, None] + yy

        # 这里是把anchor的尺寸，当做标准卷积的3x3窗口，去计算采样点位置
        # get sampling locations of anchors
        # torch.unbind 把某一个维度展开， anchors shape [H*W, 5]
        x_ctr, y_ctr, w, h, a = torch.unbind(anchors, dim=1)
        # anchor从图像上的坐标，转化为特征图上的坐标
        x_ctr, y_ctr, w, h = x_ctr / stride, y_ctr / stride, w / stride, h / stride
        cos, sin = torch.cos(a), torch.sin(a)
        # anchor的wh，除以kernel_size
        dw, dh = w / self.kernel_size[1], h / self.kernel_size[0]
        # 根据w/h对采样点进行偏移，shape [H*W, 3*3]
        x, y = dw[:, None] * xx, dh[:, None] * yy
        # 根据角度对采样点进行偏移
        xr = cos[:, None] * x - sin[:, None] * y
        yr = sin[:, None] * x + cos[:, None] * y
        # 
        x_anchor, y_anchor = xr + x_ctr[:, None], yr + y_ctr[:, None]
        # get offset filed
        offset_x = x_anchor - x_conv
        offset_y = y_anchor - y_conv
        # x, y in anchors is opposite in image coordinates,
        # so we stack them with y, x other than x, y
        offset = torch.stack([offset_y, offset_x], dim=-1)
        # NA,ks*ks*2
        offset = offset.reshape(
            anchors.size(0), -1).permute(1, 0).reshape(-1, feat_h, feat_w)
        return offset

    def forward(self, x, anchors, stride):
        # anchors shape [B,H,W,5]
        num_imgs, H, W = anchors.shape[:3]
        offset_list = [
            self.get_offset(anchors[i].reshape(-1, 5), (H, W), stride)
            for i in range(num_imgs)
        ]
        offset_tensor = torch.stack(offset_list, dim=0)
        
        x = self.relu(self.deform_conv(x, offset_tensor))
        return x
