import math
import torch
import torch.nn as nn


# 写关于ResNet相关的代码
# 用于resent18/34模型的残差块block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# 用于resent50/101/152模型的block
class BottleNeck(nn.Module):
    # 扩展，表示该残差块的输出通道数与输入通道数不同
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        '''
        inplanes:当前特征图的通道数，也即输入该残差块的特征图的通道数
        plane：该残差块的第一个卷积的输出通道数，注意不是该残差块的输出通道数，expansion*plane才是该残差块的输出通道数
        '''
        super(BottleNeck, self).__init__()
        # 第一个卷积层，减少特征图的通道数，或者保持不变
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes)
        # 第二个卷积层，减小特征图的尺寸，或者保持不变
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, # change
                    padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 第3个卷积层，扩展通道数
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # 输入量即为残差
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 是否需要下采样，在每一个layer的第一个残差块中都是需要下采样的，
        # 该layer的其他残差块不需要下采样
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    # resnet模型的基础block，以及每一层级的block的重复次数
    arch_settings = {
        "resnet18": (BasicBlock, (2, 2, 2, 2)),
        "resnet34": (BasicBlock, (3, 4, 6, 3)),
        "resnet50": (BottleNeck, (3, 4, 6, 3)),
        "resnet101": (BottleNeck, (3, 4, 23, 3)),
        "resnet152": (BottleNeck, (3, 8, 36, 3))
    }

    def __init__(self, model_name, num_classes=1000):

        assert model_name in self.arch_settings.keys(), "model don't exist !!!"
        # 获得残差块的类型，以及ResNet网络每个layer的残差块的个数
        block, layers_cfg = self.arch_settings[model_name]
        # 表示当前的特征图的通道数，会随着网络的构建过程而变化
        self.inplanes = 64
        
        super(ResNet, self).__init__()
        # 7x7的卷积，第1次下采样
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 第2次下采样
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False) # change
        self.layer1 = self._make_layer(block, 64, layers_cfg[0])

        # 第3次下采样
        self.layer2 = self._make_layer(block, 128, layers_cfg[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers_cfg[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers_cfg[3], stride=2)
        # it is slightly better whereas slower to set stride = 1
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        
        # 这里用的是f=7、s=7的平均池化，而不是全局平均池化，
        # 这是因为ResNet的输入是224，最后一层特征图的尺寸是7x7，因此f=7、s=7的平均池化也能起到全局平均池化的作用
        # 但我觉得最好还是用全局平均池化
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, plane, num_blocks, stride=1):
        '''
        plane: 该layer的残差块的第一个卷积的输出通道数
        num_blocks: 该layer的残差块的个数
        stride：该layer是否需要缩小尺寸
        '''
        downsample = None
        # 如果需要进行下采样，或者当前的特征图的通道数不等于该残差块的输出通道数，无法进行add拼接
        if stride != 1 or self.inplanes != plane * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, plane * block.expansion,
                    kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(plane * block.expansion),
            )

        layers = []
        # 该layer的第一个残差块，
        # 对layer1来讲，第一个残差块的输入特征图与输出特征图之间，没有尺寸的变化，只有通道数的变化
        # 而对layer2-4来讲，第一个残差块既有尺寸的变化，也有通道数的变化
        layers.append(block(self.inplanes, plane, stride, downsample))
        
        # 当前的特征图的通道数发生变化
        self.inplanes = plane * block.expansion
        # 该layer后续的残差块，输入特征图与输出特征图不需要调整尺寸和通道数
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, plane))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# 获得torchvision的预训练模型的下载链接
def get_torchvision_models():
    # pkautil是内置的与包管理有关的库，
    # pkgutil.walk_packages(path=path, prefix=name + ‘.’)，递归导入次文件所在文件夹中的包
    import pkgutil, torchvision, importlib
    model_urls = dict()
    for _, name, ispkg in pkgutil.walk_packages(torchvision.models.__path__):
        if ispkg:
            continue
        _zoo = importlib.import_module('torchvision.models.{}'.format(name))
        if hasattr(_zoo, 'model_urls'):
            _urls = getattr(_zoo, 'model_urls')
            model_urls.update(_urls)
    
    return model_urls

def get_dist_info():
    import torch.distributed as dist

    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    # 非分布式训练，rank默认是0，world_size默认是1
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def load_url_dist(url):
    """ In distributed setting, this function only download checkpoint at
    local rank 0 """
    
    from torch.utils import model_zoo
    import os

    rank, world_size = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    if rank == 0:
        checkpoint = model_zoo.load_url(url)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = model_zoo.load_url(url)
    return checkpoint

def load_checkpoint(checkpoint_name):
    from torch.utils import model_zoo
    # 获得torchvision中所有预训练模型的下载链接
    # {resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', }
    model_urls = get_torchvision_models()

    # 获得所需的预训练模型的下载链接
    checkpoint_url = model_urls[checkpoint_name]
    # 根据下载链接，获得预训练的权重
    # checkpoint = load_url_dist(checkpoint_url)
    checkpoint = model_zoo.load_url(checkpoint_url)

    # print(checkpoint['fc.weight'].shape)

    return checkpoint

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
    def __init__(self, backbone_name, frozen_stages=1, norm_eval=False, out_indices=(2,3,4)):
        '''
        frozen_stages : 表示冻结模型的层级，默认为1，表示ResNet网络的layer1和以前的网络均冻结参数，并且BN层药设置为eval模式
        norm_eval:在训练状态下，是否让BN层处于eval模式
        '''
        super().__init__()

        self.backbone_type = ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")
        self.backbone_name = backbone_name
        assert self.backbone_name in self.backbone_type, f"{self.backbone_name} don't exist !!!"
        
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        # 创建ResNet分类网络模型
        clasify_model = ResNet(self.backbone_name)
        # 获得预训练权重
        state_dict = load_checkpoint(self.backbone_name)
        # 导入权重
        clasify_model = load_state_dict(clasify_model, state_dict)

        # 创建用于检测网络的backbone，去掉了全局平均池化层和全连接层
        self.backbone = nn.Sequential(
            nn.Sequential(clasify_model.conv1, clasify_model.bn1,clasify_model.relu),  # C1
            nn.Sequential(clasify_model.maxpool, clasify_model.layer1),  # C2  
            clasify_model.layer2,  # C3
            clasify_model.layer3,  # C4
            clasify_model.layer4,  # C5
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
