"Interesting idea, but didn't work out."
import tensorflow as tf
from tensorflow.keras import datasets, layers

class MatMulMixer(layers.Layer):
    def __init__(self, Net=MLP):
        super().__init__()
        self.Net = Net

    def build(self, input_shape):
        SIZE = 512
        self.layer_1 = self.Net(SIZE)
        self.layer_2 = self.Net(SIZE)

    def call(self, inputs):
        h = tf.transpose(inputs, [0, 2, 1])
        h = self.layer_1(h)
        rep_1 = tf.transpose(h, [0, 2, 1])
        rep_2 = self.layer_2(inputs)
        return tf.matmul(rep_2, rep_1)
