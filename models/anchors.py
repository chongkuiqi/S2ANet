
import torch
import math


class AnchorGeneratorRotated(object):
    def __init__(self, anchor_base_size, scales, ratios=[1.0,], angles=[0,],scale_major=True):
        '''
        ratios : 长宽比, 其值不能小于1, 因为旋转框的具有周期性，容易造成混淆，因此我们定义长宽比为长边除以短边
        '''
        
        self.anchor_base_size = anchor_base_size
        self.scales = torch.Tensor(scales)
        self.ratios = torch.Tensor(ratios)
        self.angles = torch.Tensor(angles)
        
        # 长宽比必须全部大于1.0才行
        assert torch.all(self.scales >= 1.0), "ratios need bigger than 1.0"

        # 角度的单位为弧度，且范围在[-pi,pi]之间
        assert torch.all( (self.angles>-math.pi) & (self.angles<=math.pi) )


        # 默认是True
        self.scale_major = scale_major
        
        # 生成基础anchor，宽度和高度是以像素值为单位的，而不是特征图
        # shape: [num_anchor, 3(w(长边), h(短边), angle)]， num_anchor是指一个基础尺寸，不同尺度、不同长宽比、不同角度下的anchor，
        # 可以理解为一个特征层级上的anchor
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return self.base_anchors.size(0)

    def gen_base_anchors(self):
        # 方形的
        w = self.anchor_base_size
        h = self.anchor_base_size

        # 长宽比开方，也就是说，对目标框进行缩放获得所需的长宽比后，仍然保证面积不变。
        # w长边，h短边
        w_ratios = torch.sqrt(self.ratios)
        h_ratios = 1 / w_ratios
        assert self.scale_major, "AnchorGeneratorRotated only support scale-major anchors!"

        # w_ratios[:, None, None] shape : [num_ratios,1,1]
        # self.scales[None, :, None] shape : [1, num_scales, 1]
        # torch.ones_like(self.angles)[None, None, :] shape : [1,1,num_angles]
        # ws shape: (num_ratios * num_scales* num_angles)
        ws = (w * w_ratios[:, None, None] * self.scales[None, :, None] *
              torch.ones_like(self.angles)[None, None, :]).view(-1)
        hs = (h * h_ratios[:, None, None] * self.scales[None, :, None] *
              torch.ones_like(self.angles)[None, None, :]).view(-1)
        
        angles = self.angles.repeat(len(self.scales) * len(self.ratios)).view(-1)

        # shape : [num_ratios*num_scales*num_angles, 3]
        base_anchors = torch.stack((ws, hs, angles), dim=1).reshape(-1,3)

        return base_anchors

    def _make_grid(self, feat_w, feat_h):
        yv, xv = torch.meshgrid([torch.arange(feat_w), torch.arange(feat_h)])
        return torch.stack((xv, yv), 2)

    def _meshgrid(self, feat_w, feat_h):
        # xx shape : ( len(x)*len(y) ) 
        xx = feat_w.repeat(len(feat_h))
        yy = feat_h.view(-1, 1).repeat(1, len(feat_w)).view(-1)
        
        return xx, yy
        
    
    def gen_grid_anchors(self, featmap_size, featmap_stride):
        '''
        featmap_stride:该特征图的下采样倍数
        return：返回的anchor，其xywh均是以像素值为单位的
        '''
        # featmap_size*stride project it to original area
        base_anchors = self.base_anchors
        num_anchors = base_anchors.shape[0]

        # 计算网格anchor的中心点坐标
        feat_h, feat_w = featmap_size
        
        shift_x = torch.arange(0, feat_w) 
        shift_y = torch.arange(0, feat_h) 
        # shift_xx, shift_yy shape : (feat_h*feat_w)
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        # # 网格坐标乘以stride后，获得网格左上角点在图像上的坐标，以像素值为单位
        # # 另外,还要加上0.5(stride-1)，把anchor中心点放在网格中心点处
        # # 注意，是乘以(stride-1)不是stride，因为坐标起始值是以0开始的，比如8*8的图像，中心点坐标应该是(3.5,3.5)，而不是(4,4)
        shift_xx = shift_xx * featmap_stride + 0.5*(featmap_stride-1)
        shift_yy = shift_yy * featmap_stride + 0.5*(featmap_stride-1)
        # shift_xx = (shift_xx + 0.5) * featmap_stride
        # shift_yy = (shift_yy + 0.5) * featmap_stride

        shift_others = torch.zeros_like(shift_xx)

        # torch.stack会在新的维度对两个Tensor进行拼接
        # shifts shape:[feat_h*feat_w, 5]
        shifts = torch.stack(
            [shift_xx, shift_yy, shift_others, shift_others, shift_others], dim=-1)
        shifts = shifts.type_as(base_anchors)

        # 增加两个元素，为xy坐标提供位置
        base_anchors = torch.cat((torch.zeros(num_anchors, 2, device=base_anchors.device), base_anchors), dim=1)


        # 给anchors添加中心点坐标，获得网格anchors
        # base_anchors[None, :, :] shape: [1, num_anchor, 5], 
        # shifts[:, None, :] shape：[feat_h*feat_w, 1, 5]
        # grid_anchors shape : [feat_h*feat_w, num_anchor, 5]
        grid_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        
        # # reshape to (feat_h*feat_w*num_anchor, 5)
        # # first num_anchors 个 rows correspond to A anchors of (0, 0) in feature map,
        # # then (0, 1), (0, 2), ...
        # grid_anchors = grid_anchors.view(-1, 5)
        
        # reshape to [feat_h, feat_w, num_anchor, 5]
        grid_anchors = grid_anchors.view(feat_h, feat_w, num_anchors, 5)
        
        
        return grid_anchors
