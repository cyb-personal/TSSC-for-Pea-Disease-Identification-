import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 导入模型和数据加载器
from models.zfnet import ZFNetWithAttention
from data_loader import get_data_loaders, INV_CLASS_MAP

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练TSSC豌豆病害识别模型')
    parser.add_argument('--data_dir', type=str, default='./pea_disease_dataset',
                      help='数据集根目录')
    parser.add_argument('--epochs', type=int, default=50,
                      help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                      help='权重衰减')
    parser.add_argument('--img_size', type=int, default=224,
                      help='图像尺寸')
    parser.add_argument('--save_dir', type=str, default='./weights',
                      help='模型权重保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                      help='日志保存目录')
    parser.add_argument('--device', type=str, default='cuda:0',
                      help='训练设备 (cuda或cpu)')
    parser.add_argument('--resume', type=str, default=None,
                      help='从之前保存的权重继续训练')
    return parser.parse_args()

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录损失和预测结果
        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 更新进度条
        pbar.set_postfix({'loss': loss.item()})
    
    # 计算 epoch 指标
    epoch_loss = total_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, epoch_acc, epoch_f1

def evaluate(model, val_loader, criterion, device):
    """在验证集上评估模型"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算评估指标
    val_loss = total_loss / len(val_loader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    
    # 打印详细分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, 
                              target_names=INV_CLASS_MAP.values(),
                              digits=4))
    
    return val_loss, val_acc, val_f1

def main(args):
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 初始化TensorBoard
    writer = SummaryWriter(args.log_dir)
    
    # 设备配置
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    train_loader, val_loader, test_loader = get_data_loaders(
        args.data_dir, 
        batch_size=args.batch_size,
        img_size=args.img_size
    )
    
    # 初始化模型
    model = ZFNetWithAttention(num_classes=len(INV_CLASS_MAP)).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # 从检查点恢复
    best_val_f1 = 0.0
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_f1 = checkpoint['best_val_f1']
        print(f"从 {args.resume} 恢复训练，起始epoch: {start_epoch}")
    
    # 训练主循环
    print("开始训练...")
    for epoch in range(start_epoch, args.epochs):
        # 训练
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        
        # 调整学习率
        scheduler.step(val_loss)
        
        # 打印 epoch 结果
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"训练: 损失={train_loss:.4f}, 准确率={train_acc:.4f}, F1={train_f1:.4f}")
        print(f"验证: 损失={val_loss:.4f}, 准确率={val_acc:.4f}, F1={val_f1:.4f}")
        print("-" * 80)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            checkpoint_path = os.path.join(args.save_dir, 'best_tssc.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1,
            }, checkpoint_path)
            print(f"保存最佳模型到 {checkpoint_path} (F1: {best_val_f1:.4f})")
        
        # 每10个epoch保存一次 checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'tssc_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1,
            }, checkpoint_path)
    
    # 训练结束后在测试集上评估
    print("训练完成，在测试集上评估最佳模型...")
    best_model_path = os.path.join(args.save_dir, 'best_tssc.pth')
    model.load_state_dict(torch.load(best_model_path, map_location=device)['model_state_dict'])
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)
    print(f"测试集结果: 损失={test_loss:.4f}, 准确率={test_acc:.4f}, F1={test_f1:.4f}")
    
    # 保存测试集结果
    with open(os.path.join(args.log_dir, 'test_results.txt'), 'w') as f:
        f.write(f"测试集损失: {test_loss:.4f}\n")
        f.write(f"测试集准确率: {test_acc:.4f}\n")
        f.write(f"测试集F1分数: {test_f1:.4f}\n")
    
    writer.close()

if __name__ == '__main__':
    args = parse_args()
    main(args)
