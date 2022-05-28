import torch.nn as nn
import torch.nn.functional as F

from models.init_weights import xavier_init
class FPN(nn.Module):
    def __init__(self, in_channels=[512, 1024, 2048], out_channels=256, num_outs=5):
        '''
        in_channels: 输入的特征图的通道数列表
        out_channels: 输出特征图的通道数
        '''
        super().__init__()

        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        # 额外的特征层级 
        self.num_extra_levels = self.num_outs - self.num_ins
        if self.num_extra_levels < 1:
            self.num_extra_levels = 0

        # down-up的顺序
        self.lateral_convs = nn.ModuleList()
        # down-up的顺序
        self.fpn_convs = nn.ModuleList()


        for i in range(self.num_ins):
            # 横向连接将backbone各个层级的特征图的通道数，调整为256
            lateral_conv = nn.Conv2d(self.in_channels[i], self.out_channels, 
                kernel_size=(1,1), stride=(1,1), padding=0, bias=True)
            
            # fpn连接的目的，是对up-down融合后的特征图进一步进行处理
            fpn_conv = nn.Conv2d(self.out_channels, self.out_channels, 
                kernel_size=(3,3), stride=(1,1), padding=1, bias=True)
            
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)
        
        
        if self.num_extra_levels > 0:
            for i in range(self.num_extra_levels):
                # 由C5进行s=2的下采样卷积得到P6，通道数要由2048调整为256
                # 由P6进行s=2的下采样卷积得到P7，通道数不需调整
                if i == 0:
                    fpn_conv = nn.Conv2d(self.in_channels[-1], self.out_channels, 
                        kernel_size=(3,3), stride=(2,2), padding=1, bias=True)
                else:
                    fpn_conv = nn.Conv2d(self.out_channels, self.out_channels, 
                        kernel_size=(3,3), stride=(2,2), padding=1, bias=True)
                
                self.fpn_convs.append(fpn_conv)   
        
        # 初始化
        self.init_weights()

    def init_weights(self):
        # print("1")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
    
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # （1）建立横向连接，对齐通道数
        # build laterals
        temporaries = [
            lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # （2）建立top-down路径，元素点加
        # build top-down path
        # range(start, stop[, step])，不包括stop
        for i in range(self.num_ins - 1, 0, -1):
            # 这种叠加方法，可以保证最底层的特征图，能够获得上面所有层的特征信息，而不只是相邻的上一层
            temporaries[i - 1] += F.interpolate(
                temporaries[i], scale_factor=2, mode='nearest')
        
        # （3）获得输出，即P3-P5
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](temporaries[i]) for i in range(self.num_ins)
        ]
        # part 2: add extra levels
        if self.num_extra_levels > 0:
            
            for i in range(self.num_extra_levels):
                # 使用C5获得P6
                if i == 0:
                    outs.append(self.fpn_convs[self.num_ins](inputs[-1]))
                else:
                    outs.append(self.fpn_convs[self.num_ins+i](outs[-1]))
        
        return tuple(outs)


class PAN(FPN):
    def __init__(self, in_channels=[512, 1024, 2048], out_channels=256, num_outs=5):
        '''
        in_channels: 输入的特征图的通道数列表
        out_channels: 输出特征图的通道数
        '''
        super().__init__(in_channels=in_channels, out_channels=out_channels, num_outs=num_outs)

        # PAN的bottom-up路径的下采样操作，是由stride=2的卷积层实现的
        self.pan_downsample = nn.ModuleList()
        self.pan_out_convs = nn.ModuleList()
        for i in range(self.num_ins - 1):

            self.pan_downsample.append(
                nn.Sequential(
                    nn.Conv2d(self.out_channels, self.out_channels, kernel_size=(3,3), stride=(2,2), padding=1, bias=True),
                    nn.ReLU(inplace=True)
                )
            )

            self.pan_out_convs.append(
                nn.Sequential(
                    nn.Conv2d(self.out_channels, self.out_channels, kernel_size=(3,3), stride=(1,1), padding=1, bias=True),
                    nn.ReLU(inplace=True)
                )
            )
        
        if self.num_extra_levels > 0:
            for i in range(self.num_extra_levels):
                self.pan_out_convs.append(
                    nn.Sequential(
                        nn.Conv2d(self.out_channels, self.out_channels, kernel_size=(3,3), stride=(1,1), padding=1, bias=True),
                        nn.ReLU(inplace=True)
                    )
                ) 

        # 在PAN结构中我们没有进行权重初始化，因为FPN的卷积已经采用xavier初始化了，
        # 而PAN独有的卷积默认采用的是he初始化方法，对采用ReLu激活函数的卷积来讲，he初始化是合适的


    
    def forward(self, inputs):
        
        # 进行FPN结构的前向传播
        outs = super().forward(inputs)

        outs = list(outs)

        # 建立bottom-up连接
        # range(start, stop[, step])，不包括stop
        # 需要注意的是，PAN的bottomup操作，是先计算出当前层级的最终特征图，然后将最终特征图送入上一层级进行处理,这跟FPN不一样
        for i in range(1, self.num_ins):
            outs[i] += self.pan_downsample[i-1](outs[i-1])
            # 对当前融合特征进行卷积，得到最终的特征图
            outs[i] = self.pan_out_convs[i-1](outs[i])
        
        if self.num_extra_levels > 0:
            for i in range(self.num_extra_levels):
                idx = self.num_ins + i
                outs[idx] = self.pan_out_convs[idx-1](outs[idx])

        return tuple(outs)
