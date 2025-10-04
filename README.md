以下是调整格式后的 README.md 内容，优化了标题层级、列表缩进、链接样式及排版一致性：


# TSSC-for-Pea-Disease-Identification

> 官方 PyTorch 实现 | 论文处于投刊阶段，标题：《TSSC: A New Deep Learning Model for Accurate Pea Leaf Disease Identification》  


> 提出时空尺度通道网络（TSSC）模型，实现五类豌豆常见病害与健康状态的高精度识别，助力农业病害智能化诊断。  


## 1. 研究背景与模型定位

豌豆作为重要豆科作物，其叶片病害（如白粉病、锈病、霜霉病等）易导致产量下降，传统人工检测存在效率低、依赖经验的问题。  

本文提出**时空尺度通道网络（TSSC）**，通过创新的多维度特征融合与跨尺度建模机制，解决豌豆病害“多类别区分难、复杂背景干扰大”的问题，最终在五类豌豆病害数据集上实现优异的分类性能，为农业病害自动化诊断提供高效解决方案。  


## 2. TSSC 核心创新点

1. **三层特征交互机制**：  
   - 空间层：采用自适应区域注意力（Adaptive Region Attention），精准定位不同病害的典型病灶区域（如斑点、霉层、卷曲边缘）；  
   - 尺度层：应用跨尺度特征聚合（Cross-Scale Feature Aggregation），融合多分辨率特征图，适配不同大小的病害表现；  
   - 通道层：使用动态通道选择（Dynamic Channel Selection），强化病害关键特征通道的响应，抑制无关背景干扰。  

2. **时空特征增强模块**：  
   通过模拟病害发展时序特性，对静态图像进行隐式时序特征建模，提升对早期微小病害的识别能力。  

3. **多类别适配优化**：  
   针对五类病害的细粒度差异，设计类别感知损失函数，在保证高准确率的同时降低类别混淆，尤其提升相似病害（如锈病与叶斑病）的区分度。  


## 3. 实验数据集：五类豌豆病害数据集

### 3.1 数据集概况

本研究基于**五类豌豆病害识别数据集**，包含五种常见豌豆叶片状态，数据集存储于百度网盘，需自行下载后使用：  

| 数据集名称 | 包含类别 | 图像总数 | 图像分辨率 | 数据分布（训练:验证:测试） |
|------------|------------------------------|----------|------------|-----------------------|
| 五类豌豆数据集 | 白粉病、锈病、霜霉病、叶斑病 + 健康叶片 | 12,000+ | 统一 resize 至 256×256 | 7:1:2（通过代码自动划分） |


### 3.2 数据集获取与结构

1. **下载链接**：  
   百度网盘链接：[https://pan.baidu.com/s/18FxZMhVcK-5hRwAhoJS8mQ](https://pan.baidu.com/s/18FxZMhVcK-5hRwAhoJS8mQ)  
   提取码: bq9g（复制内容后打开百度网盘手机App操作更方便）  


2. **文件夹组织**（下载后解压至项目根目录，结构如下）：  
```
pea_disease_dataset/
├── 白粉病/       # 豌豆白粉病叶片图像
├── 锈病/         # 豌豆锈病叶片图像
├── 霜霉病/       # 豌豆霜霉病叶片图像
├── 叶斑病/       # 豌豆叶斑病叶片图像
└── 健康叶片/     # 健康豌豆叶片图像
```


## 4. 实验环境配置

### 4.1 依赖安装

推荐使用 Anaconda 创建虚拟环境，确保依赖版本匹配（避免兼容性问题）：  

```bash
# 1. 创建并激活虚拟环境
conda create -n tssc-pea python=3.10
conda activate tssc-pea

# 2. 安装PyTorch与TorchVision（需适配CUDA版本，示例为CUDA 12.1）
pip3 install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu121

# 3. 安装其他依赖库
pip install numpy~=2.0.2 matplotlib~=3.9.4 opencv-python~=4.12.0.88
pip install pandas~=2.3.1 pillow~=11.2.1 torchviz~=0.0.3 xlwt~=1.3.0
pip install tqdm~=4.67.1 timm~=1.0.15 scikit-learn~=1.5.1
```


### 4.2 硬件要求

- **GPU**：推荐 NVIDIA GPU（显存≥10GB，如RTX 3060/4060，支持CUDA 11.8+），训练耗时约3-4小时（60轮）；  
- **CPU**：支持推理测试，但训练耗时显著增加（约25-30小时），不推荐用于训练。  


## 5. 实验结果

### 5.1 核心指标对比（五类豌豆数据集）

TSSC 与主流深度学习模型在五类豌豆病害识别任务上的性能对比如下，模型在多类别区分能力上表现更优：  

| 模型 | 分类准确率（Accuracy） | 宏平均F1值（Macro-F1） | 计算量（FLOPs） | 参数量（M） |
|--------------------|------------------------|------------------------|-----------------|-------------|
| Vision Transformer（ViT-Base） | 95.27% | 94.83% | 17.6G | 86.8 |
| EfficientNet-B0 | 93.15% | 92.67% | 0.39G | 5.3 |
| Swin Transformer（Tiny） | 94.08% | 93.52% | 4.5G | 28.2 |
| **TSSC（本文）** | **98.65%** | **98.42%** | **12.3G** | **14.2** |


> 注：准确率与宏平均F1值为5次实验的平均值，确保结果稳定性；计算量较ViT-Base降低约30%。  


## 6. 代码使用说明

### 6.1 模型训练

运行 `train.py` 脚本启动训练，支持通过参数调整训练配置，示例命令：  

```bash
python train.py \
  --data_dir ./pea_disease_dataset \  # 数据集根目录（解压后的路径）
  --epochs 60 \
  --batch_size 16 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --save_dir ./weights \
  --device cuda:0 \
  --log_interval 10  # 每10个batch打印一次训练日志
```


#### 关键参数说明：

| 参数名 | 含义 | 默认值 |
|-----------------|---------------------------------------|-----------------|
| `--data_dir` | 数据集根目录路径 | `./pea_disease_dataset` |
| `--epochs` | 训练轮数 | 60 |
| `--batch_size` | 批次大小（根据GPU显存调整，8/16/32） | 16 |
| `--lr` | 初始学习率 | 1e-4 |
| `--save_dir` | 训练权重保存目录 | `./weights` |
| `--device` | 训练设备（`cuda:0` 或 `cpu`） | `cuda:0` |


#### 训练输出：

- 训练过程中，模型会自动保存**验证集宏平均F1值最高**的权重至 `--save_dir` 目录，文件名为 `best_tssc.pth`；  
- 训练日志（损失值、准确率、F1值）会实时打印，并保存至 `train_log.txt`。  


### 6.2 模型预测

使用训练好的权重进行单张豌豆叶片图像预测，运行 `predict.py` 脚本，示例命令：  

```bash
python predict.py \
  --image_path ./examples/pea_mildew.jpg \  # 输入图像路径（示例图像存于examples/）
  --weight_path ./weights/best_tssc.pth \  # 预训练权重路径
  --device cuda:0
```


#### 预测输出示例：

```
输入图像路径：./examples/pea_mildew.jpg
预测类别：豌豆霜霉病
置信度：0.9987
```


## 7. 项目文件结构

```
tssc-for-pea-disease-identification/
├── pea_disease_dataset/  # 五类豌豆病害数据集（需从百度网盘下载）
├── examples/             # 预测示例图像（如rust_example.png）
├── models/               # 模型定义文件夹
│   ├── Tssc.py           # TSSC核心代码（含特征交互、尺度建模模块）
│   ├── se_module.py      # 实现 Squeeze-and-Excitation 注意力机制
│   ├── split_attention.py    # 实现split注意力机制
│   ├── three_neighbor_attention.py   # 实现三邻域通道注意力机制
├── dataset/              # 数据处理文件夹
│   ├── data_loader.py    # 数据集加载与预处理（自动划分训练/验证/测试集）
├── train.py              # 模型训练脚本
├── predict.py            # 模型预测脚本
└── README.md             # 项目说明文档（本文档）
```


## 8. 已知问题与注意事项

1. **数据集适配**：当前模型与权重仅针对上述五类豌豆状态，若新增病害类别，需补充数据集并重新训练；  
2. **图像分辨率**：输入图像会自动resize至256×256，若原始图像分辨率过低（<128×128），可能导致特征丢失，建议输入图像分辨率≥256×256；  
3. **数据集下载**：若百度网盘链接失效，请联系作者获取最新链接；  
4. **CUDA版本问题**：若安装PyTorch时出现CUDA不兼容，可替换为CPU版本（需将所有脚本的`--device`改为`cpu`），但训练效率会大幅下降。  


## 9. 引用与联系方式

### 9.1 引用方式

论文处于投刊阶段，正式发表后将更新BibTeX引用格式，当前可临时引用：  

```bibtex
@article{tssc_pea_disease,
  title={TSSC: A New Deep Learning Model for Accurate Pea Leaf Disease Identification},
  author={[作者姓名，待发表时补充]},
  journal={[期刊名称，待录用后补充]},
  year={2024},
  note={Manuscript submitted for publication}
}
```


### 9.2 联系方式

若遇到代码运行问题或学术交流需求，请联系：  
- 邮箱：changyibu@huuc.edu.cn  
- GitHub Issue：直接在本仓库提交Issue，会在1-3个工作日内回复。  

