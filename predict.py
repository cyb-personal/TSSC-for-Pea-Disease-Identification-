import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# 导入模型和类别映射
from models.zfnet import ZFNetWithAttention
from data_loader import INV_CLASS_MAP

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用TSSC模型预测豌豆病害')
    parser.add_argument('--image_path', type=str, required=True,
                      help='输入图像路径')
    parser.add_argument('--model_path', type=str, default='./weights/best_tssc.pth',
                      help='模型权重路径')
    parser.add_argument('--img_size', type=int, default=224,
                      help='图像尺寸')
    parser.add_argument('--device', type=str, default='cuda:0',
                      help='预测设备 (cuda或cpu)')
    return parser.parse_args()

def load_model(model_path, num_classes, device):
    """加载模型"""
    model = ZFNetWithAttention(num_classes=num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # 处理不同的 checkpoint 格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model

def preprocess_image(image_path, img_size):
    """预处理输入图像"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 添加批次维度
    return image

def predict(args):
    """预测豌豆病害类别"""
    # 设备配置
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 检查图像文件是否存在
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"图像文件不存在: {args.image_path}")
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
    
    # 加载模型
    num_classes = len(INV_CLASS_MAP)
    model = load_model(args.model_path, num_classes, device)
    print(f"成功加载模型: {args.model_path}")
    
    # 预处理图像
    image = preprocess_image(args.image_path, args.img_size).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_class_idx = np.argmax(probabilities)
        pred_class_name = INV_CLASS_MAP[pred_class_idx]
        confidence = probabilities[pred_class_idx]
    
    # 输出结果
    print(f"\n===== 预测结果 =====")
    print(f"输入图像: {args.image_path}")
    print(f"预测类别: {pred_class_name}")
    print(f"置信度: {confidence:.4f}")
    print("\n类别概率分布:")
    for idx, prob in enumerate(probabilities):
        print(f"  {INV_CLASS_MAP[idx]}: {prob:.4f}")
    
    return pred_class_name, confidence

if __name__ == '__main__':
    args = parse_args()
    predict(args)
