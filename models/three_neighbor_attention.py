import torch
import torch.nn as nn
import torch.nn.functional as F

class ThreeNeighborChannelAttention(nn.Module):
    """
    三邻域通道注意力模块
    考虑每个通道与其前后相邻通道的关系来计算注意力权重
    """
    def __init__(self, channel, reduction=16):
        super(ThreeNeighborChannelAttention, self).__init__()
        self.channel = channel
        self.reduction = reduction
        
        # 用于捕获通道间关系的卷积层
        self.conv = nn.Conv1d(3, 1, kernel_size=1, stride=1, padding=0)
        
        # 瓶颈结构
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        
        # 全局平均池化
        y = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)  # 形状: [b, c]
        
        # 构建三邻域特征 (当前通道, 前一个通道, 后一个通道)
        # 处理边界情况
        prev_channel = torch.cat([y[:, 0:1], y[:, :-1]], dim=1)  # 前一个通道，第一个通道的前邻是自身
        next_channel = torch.cat([y[:, 1:], y[:, -1:]], dim=1)  # 后一个通道，最后一个通道的后邻是自身
        
        # 堆叠形成三邻域特征 [b, 3, c]
        neighbor_feat = torch.stack([y, prev_channel, next_channel], dim=1)
        
        # 捕获三邻域关系
        attention = self.conv(neighbor_feat).squeeze(1)  # [b, c]
        
        # 通过全连接层得到最终注意力权重
        attention = self.fc(attention).view(b, c, 1, 1)
        
        # 应用注意力权重
        return x * attention.expand_as(x)
