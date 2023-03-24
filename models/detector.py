# 检测器

import torch.nn as nn

from models.backbone import DetectorBackbone
from models.neck import FPN
from models.head import S2ANetHead

class S2ANet(nn.Module):
    def __init__(self, backbone_name="resnet50", num_classes=15):
        super().__init__()
        
        # # 每个特征层级的下采样次数
        self.stride = (8, 16, 32, 64, 128)
        # 用于检测的每个特征层级的下采样次数
        # self.stride = [8, 16, 32]
        self.nl = len(self.stride)  # 检测层的个数，即neck网络输出的特征层级的个数

        # backbone输出C3、C4、C5三个特征图
        self.backbone = DetectorBackbone(backbone_name)

        self.neck = FPN(num_outs=self.nl)

        
        self.head = S2ANetHead(num_classes=num_classes, featmap_strides=self.stride)


    def forward(self, imgs, targets=None, post_process=False):

        outs = self.backbone(imgs)
        outs = self.neck(outs)


        imgs_size = imgs.shape[-2:]
        results = self.head(outs, targets=targets, imgs_size=imgs_size, post_process=post_process)

        return results

