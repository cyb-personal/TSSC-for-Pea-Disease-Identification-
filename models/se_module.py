import tensorflow as tf
from tensorflow.keras import layers

class SEModule(layers.Layer):
    """Squeeze-and-Excitation 通道注意力模块"""
    def __init__(self, channels, ratio=16, **kwargs):
        super(SEModule, self).__init__(**kwargs)
        self.channels = channels
        self.fc1 = layers.Dense(channels // ratio, activation="relu", use_bias=False)
        self.fc2 = layers.Dense(channels, activation="sigmoid", use_bias=False)

    def call(self, inputs):
        # 1. 全局平均池化（Squeeze）
        se = layers.GlobalAveragePooling2D()(inputs)
        # 2. 通道权重学习（Excitation）
        se = self.fc1(se)
        se = self.fc2(se)
        # 3. 权重广播并作用于输入
        return inputs * tf.reshape(se, (-1, 1, 1, self.channels))