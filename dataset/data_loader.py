import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# 定义类别映射（五类豌豆状态）
CLASS_MAP = {
    "白粉病": 0,
    "锈病": 1,
    "霜霉病": 2,
    "叶斑病": 3,
    "健康叶片": 4
}

# 反转类别映射（用于预测时转换回类别名称）
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

class PeaDiseaseDataset(Dataset):
    """豌豆病害数据集加载器"""
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []
        
        # 加载数据集
        self._load_dataset()
        
    def _load_dataset(self):
        """加载图像路径和对应的标签"""
        for class_name, class_idx in CLASS_MAP.items():
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                raise ValueError(f"类别文件夹不存在: {class_dir}")
                
            # 获取该类别下所有图像
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            # 划分训练/验证/测试集 (7:1:2)
            total = len(image_files)
            train_split = int(0.7 * total)
            val_split = int(0.8 * total)
            
            if self.split == 'train':
                selected_files = image_files[:train_split]
            elif self.split == 'val':
                selected_files = image_files[train_split:val_split]
            elif self.split == 'test':
                selected_files = image_files[val_split:]
            else:
                selected_files = image_files
                
            # 保存图像路径和标签
            for file in selected_files:
                self.images.append(os.path.join(class_dir, file))
                self.labels.append(class_idx)
                
        print(f"加载{self.split}集完成: {len(self.images)}张图像, {len(CLASS_MAP)}个类别")
                
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"无法打开图像 {img_path}: {str(e)}")
        
        # 应用数据变换
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_loaders(data_dir, batch_size=16, img_size=224):
    """获取训练、验证和测试数据加载器"""
    # 定义数据变换
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = PeaDiseaseDataset(
        data_dir, split='train', transform=train_transform
    )
    val_dataset = PeaDiseaseDataset(
        data_dir, split='val', transform=val_test_transform
    )
    test_dataset = PeaDiseaseDataset(
        data_dir, split='test', transform=val_test_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    return train_loader, val_loader, test_loader
