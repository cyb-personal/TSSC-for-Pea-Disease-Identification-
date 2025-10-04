import torch
import torch.nn as nn
import torch.nn.functional as F

class SplitAttention(nn.Module):
    """
    分裂注意力机制模块
    参考ResNeSt中的实现思路，将特征分组并分别计算注意力
    """
    def __init__(self, channel, num_groups=2, reduction=16):
        super(SplitAttention, self).__init__()
        self.channel = channel
        self.num_groups = num_groups
        self.group_channels = channel // num_groups
        
        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 每个组的注意力计算
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, h, w = x.size()
        
        # 全局平均池化
        y = self.avg_pool(x).view(b, c)  # [b, c]
        
        # 计算注意力权重
        y = self.fc1(y)  # [b, c//reduction]
        y = self.relu(y)
        y = self.fc2(y)  # [b, c]
        
        # 重塑以进行分组注意力计算
        y = y.view(b, self.num_groups, self.group_channels)  # [b, num_groups, group_channels]
        y = self.softmax(y)  # 在组维度上应用softmax
        y = y.view(b, c, 1, 1)  # [b, c, 1, 1]
        
        # 应用注意力权重
        return x * y.expand_as(x)
