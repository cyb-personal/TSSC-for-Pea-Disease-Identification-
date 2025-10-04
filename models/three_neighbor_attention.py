import tensorflow as tf
from tensorflow.keras import layers

class ThreeNeighborChannelAttention(layers.Layer):
    """三邻域通道注意力模块（考虑相邻通道关联性）"""
    def __init__(self, channels, ratio=16, **kwargs):
        super(ThreeNeighborChannelAttention, self).__init__(**kwargs)
        self.channels = channels
        self.fc1 = layers.Dense(channels // ratio, activation="relu", use_bias=False)
        self.fc2 = layers.Dense(channels, activation="sigmoid", use_bias=False)

    def call(self, inputs):
        # 1. 计算左、右邻域通道并取平均
        left_neighbor = tf.pad(inputs[:, :, :, :-1], [[0, 0], [0, 0], [0, 0], [1, 0]])
        right_neighbor = tf.pad(inputs[:, :, :, 1:], [[0, 0], [0, 0], [0, 0], [0, 1]])
        neighbor_avg = (left_neighbor + inputs + right_neighbor) / 3
        # 2. 通道权重学习
        se = layers.GlobalAveragePooling2D()(neighbor_avg)
        se = self.fc1(se)
        se = self.fc2(se)
        # 3. 权重广播并作用于输入
        return inputs * tf.reshape(se, (-1, 1, 1, self.channels))