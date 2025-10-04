import torch
import torch.nn as nn
from se_module import SEModule
from three_neighbor_attention import ThreeNeighborChannelAttention
from split_attention import SplitAttention

class ZFNetWithAttention(nn.Module):
    """
    加入注意力机制的ZFNet模型
    在不同卷积块后分别应用三种不同的注意力机制
    """
    def __init__(self, num_classes=1000):
        super(ZFNetWithAttention, self).__init__()
        
        # 特征提取层
        self.features = nn.Sequential(
            # 第一个卷积块 - 后接SE模块
            nn.Conv2d(3, 48, kernel_size=7, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            SEModule(48),  # 应用SE注意力模块
            
            # 第二个卷积块 - 后接三邻域通道注意力
            nn.Conv2d(48, 128, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ThreeNeighborChannelAttention(128),  # 应用三邻域通道注意力
            
            # 第三个卷积块
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # 第四个卷积块
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # 第五个卷积块 - 后接分裂注意力
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            SplitAttention(128, num_groups=4),  # 应用分裂注意力机制
        )
        
        # 分类器层
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        
        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 128 * 6 * 6)  # 展平特征图
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层使用Kaiming正态分布初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全连接层使用特定正态分布初始化
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# 测试模型
if __name__ == '__main__':
    # 创建模型实例
    model = ZFNetWithAttention(num_classes=1000)
    
    # 生成随机输入 (batch_size=2, channels=3, height=224, width=224)
    input_tensor = torch.randn(2, 3, 224, 224)
    
    # 前向传播
    output = model(input_tensor)
    
    # 打印输出形状
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")  # 应输出 torch.Size([2, 1000])
