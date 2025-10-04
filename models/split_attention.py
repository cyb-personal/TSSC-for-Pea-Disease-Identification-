import tensorflow as tf
from tensorflow.keras import layers

class SplitAttention(layers.Layer):
    """分裂注意力模块（分组通道注意力）"""
    def __init__(self, channels, num_groups=4, ratio=16, **kwargs):
        super(SplitAttention, self).__init__(**kwargs)
        self.channels = channels
        self.num_groups = num_groups
        self.group_channels = channels // num_groups
        self.fc1 = layers.Dense(self.group_channels // ratio, activation="relu", use_bias=False)
        self.fc2 = layers.Dense(self.group_channels, activation="sigmoid", use_bias=False)

    def call(self, inputs):
        batch, h, w, _ = inputs.shape
        # 1. 通道分组
        grouped = tf.reshape(inputs, (-1, h, w, self.num_groups, self.group_channels))
        # 2. 组内全局池化 + 权重学习
        se = layers.GlobalAveragePooling2D()(grouped)
        se = self.fc1(se)
        se = self.fc2(se)
        # 3. 应用注意力并重组通道
        output = grouped * tf.reshape(se, (-1, 1, 1, self.num_groups, self.group_channels))
        return tf.reshape(output, (-1, h, w, self.channels))