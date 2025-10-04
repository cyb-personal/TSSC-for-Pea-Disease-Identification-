import torch
import torch.nn as nn
import torch.nn.functional as F

class SEModule(nn.Module):
    """
    Squeeze-and-Excitation Network模块
    参考论文: https://arxiv.org/abs/1709.01507
    """
    def __init__(self, channel, reduction=16):
        super(SEModule, self).__init__()
        # Squeeze操作: 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Excitation操作: 两个全连接层组成的瓶颈结构
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: 全局平均池化得到1x1xC的特征
        y = self.avg_pool(x).view(b, c)
        # Excitation: 通过全连接层得到通道注意力权重
        y = self.fc(y).view(b, c, 1, 1)
        # 应用注意力权重到输入特征
        return x * y.expand_as(x)
