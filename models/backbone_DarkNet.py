import math
import torch
import torch.nn as nn


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

# Conv + BN + LeakyReLu
class CBL(nn.Module):
    # Standard convolution
    def __init__(self, in_chn, out_chn, kernel_size=1, stride=1, padding=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(in_chn, out_chn, kernel_size, stride=stride, padding=autopad(kernel_size, padding), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_chn)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# DarkNet的残差块和ResNet不一样，add操作在第二个卷积的激活函数之后，而不是ResNet中的在第二个卷积的激活函数之前
class DarknetResblock(nn.Module):
    '''
        DarkNet的残差模块还有一个特点，就是第一个卷积层的输出通道数，是输入特征图通道数的一半，而第二个卷积层恢复为输入通道数
    '''

    def __init__(self, in_chn, reduction_ratio=2):
        super(DarknetResblock, self).__init__()
        
        # 第一个卷积层的输出通道数为输入通道数的一半
        tep_chn = in_chn // reduction_ratio

        # 两个卷积均保证图像尺寸不变
        self.CBL1 = CBL(in_chn, tep_chn, kernel_size=1, stride=1, padding=0)
        self.CBL2 = CBL(tep_chn, in_chn, kernel_size=3, stride=1, padding=1)


    def forward(self, x):

        out = x + self.CBL2(self.CBL1(x))

        return out


class DarkNet50(nn.Module):
    # resnet模型的基础block，以及每一层级的block的重复次数

    def __init__(self, num_classes=1000):
        
        super(DarkNet50, self).__init__()

        self.CBL = CBL(3, 32, kernel_size=3, stride=1, padding=1)
        # 第一次下采样
        self.layer1 = self._make_layer(in_chn=32, num_blocks=1)
        # 第二次下采样
        self.layer2 = self._make_layer(in_chn=64, num_blocks=2)
        self.layer3 = self._make_layer(in_chn=128, num_blocks=8)
        self.layer4 = self._make_layer(in_chn=256, num_blocks=8)
        self.layer5 = self._make_layer(in_chn=512, num_blocks=4)


        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, in_chn, num_blocks, chn_expansion=2):
        '''
        plane: 该layer的残差块的第一个卷积的输出通道数
        num_blocks: 该layer的残差块的个数
        stride：该layer是否需要缩小尺寸
        '''
        layers = []
        # DarkNet的每个阶段的第一个卷积，进行2倍下采样，并且通道数扩展2倍
        chn = in_chn * chn_expansion
        layers.append(CBL(in_chn, chn, kernel_size=3, stride=2, padding=1))

        # 叠加残差块
        for i in range(num_blocks):
            layers.append(DarknetResblock(chn))

        return nn.Sequential(*layers)


    def forward(self, x):
        
        x = self.CBL(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x




# 把预训练模型中的参数导入到，建立的模型中
def load_state_dict(model, state_dict):
    from utils.general import intersect_dicts
    # 注意，这里用model.state_dict()而不能用model.named_parameters()
    # named_parameters只包括可以训练的参数，而state_dict还包括不可训练的参数，即BN层的running_mean、running_var
    csd = intersect_dicts(state_dict, model.state_dict(), exclude=[])  # intersect
    model.load_state_dict(csd, strict=True)  # load
    # 对ResNet50模型来讲，这里会少53个参数，全部是num_batches_tracked参数，预训练模型中没有该参数
    print(f'Transferred {len(csd)}/{len(model.state_dict())}')  # report
    
    # k1 = state_dict.keys()
    # k2 = model.state_dict().keys()

    # k_not_have = []
    # for k in k2:
    #     if k not in k1:
    #         k_not_have.append(k)

    # print(k_not_have)
    # print(len(k_not_have))
    return model





class DetectorBackbone(nn.Module):
    def __init__(self, backbone_name,frozen_stages=1, norm_eval=False, out_indices=(2,3,4)):
        '''
        frozen_stages : 表示冻结模型的层级，默认为1，表示ResNet网络的layer1和以前的网络均冻结参数，并且BN层药设置为eval模式
        norm_eval:在训练状态下，是否让BN层处于eval模式
        '''
        super().__init__()
        
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        # 创建ResNet分类网络模型
        clasify_model = DarkNet50()

        # 使用DarkNet官方提供的在ImageNet上的预训练模型
        state_dict = torch.load("/home/lab/ckq/yolov3/offical_darknet53_conv_74.pt", map_location=torch.device("cpu"))
        ## 经过验证，预训练模型的参数顺序与我们重新实现的参数的顺序完全一致，因此只需要改变一下参数的名称就好了
        
        clasify_model_state_dict = clasify_model.state_dict()
        assert len(state_dict) == len(clasify_model_state_dict)

        csd = {k2:v1 for (k1,v1), (k2,v2) in zip(state_dict.items(), clasify_model_state_dict.items())}

        clasify_model.load_state_dict(csd, strict=True)  # load

        print(f'Transferred {len(csd)}/{len(clasify_model_state_dict)}')  # report

        # 创建用于检测网络的backbone，去掉了全局平均池化层和全连接层
        self.backbone = nn.Sequential(
            nn.Sequential(clasify_model.CBL, clasify_model.layer1),  # C1
            clasify_model.layer2,  # C2  
            clasify_model.layer3,  # C3
            clasify_model.layer4,  # C4
            clasify_model.layer5,  # C5
        )

        self._freeze_stages()


        # 需要保留的特征层级：C3-C5
        self.out_indices = out_indices

        # 销毁分类模型
        del clasify_model
    

    # 冻结模型参数，并设置为eval模式，保证BN层不会更新参数，也不会统计batch的个数，也不会更新running_mean和running_var
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i,m in enumerate(self.backbone):
                if i <= self.frozen_stages:
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False
    

    # 重写model.train()函数，保证在训练状态下，参数冻结部分的模型处于eval模式，并且BN层不会更新、不会统计
    def train(self, mode=True):
        # 先让整个模型处于训练状态
        super().train(mode)
        # 然后冻结模型，使得冻结部分处于eval模式
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        

    def forward(self, x):

        outs = []
        for i,m in enumerate(self.backbone):
            x = m(x)
            if i in self.out_indices:
                outs.append(x)
        
        return tuple(outs)
