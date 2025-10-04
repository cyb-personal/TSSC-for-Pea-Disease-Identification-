import tensorflow as tf
from tensorflow.keras import layers, models

# 从各模块文件导入注意力类
from se_module import SEModule
from split_attention import SplitAttention
from three_neighbor_attention import ThreeNeighborChannelAttention

def build_tssc_model(num_classes=5):
    """TSSC模型"""
    inputs = layers.Input(shape=(400, 400, 3), name="input")
    x = inputs

    # 第1卷积块 + SE注意力
    x = layers.Conv2D(
        filters=64,
        kernel_size=(4, 4),
        strides=(2, 2),
        padding="same",
        kernel_initializer="he_normal",
        name="conv1"
    )(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="pool1")(x)
    x = SEModule(channels=64, name="se_attention")(x)

    # 第2卷积块 + 三邻域注意力
    x = layers.Conv2D(
        filters=128,
        kernel_size=(2, 2),
        strides=(2, 2),
        padding="valid",
        kernel_initializer="he_normal",
        name="conv2"
    )(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), name="pool2")(x)
    x = ThreeNeighborChannelAttention(channels=128, name="three_neighbor_attention")(x)

    # 第3卷积块
    x = layers.Conv2D(
        filters=256,
        kernel_size=(5, 5),
        strides=(2, 2),
        padding="same",
        kernel_initializer="he_normal",
        name="conv3"
    )(x)
    x = layers.ReLU(name="relu3")(x)

    # 第4卷积块 + 分裂注意力
    x = layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="valid",
        kernel_initializer="he_normal",
        name="conv4"
    )(x)
    x = layers.ReLU(name="relu4")(x)
    x = SplitAttention(channels=128, name="split_attention")(x)

    # 分类头（全局池化 + 全连接）
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(1024, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="TSSC")

# 测试代码（可选，用于验证模型结构）
if __name__ == "__main__":
    model = build_tssc_model(num_classes=5)
    model.summary()  # 打印模型结构
    test_input = tf.random.normal(shape=(2, 400, 400, 3))  # 模拟2个样本输入
    test_output = model(test_input)
    print(f"输入形状: {test_input.shape}, 输出形状: {test_output.shape}")