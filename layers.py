import tensorflow as tf
from tensorflow.keras.layers import Layer


class MaxPoolingWithArgmax2D(Layer):
    def init(self, pool_size=(2, 2), name=None):
        super(MaxPoolingWithArgmax2D, self).init(name=name)
        self.pool_size = pool_size

    def call(self, inputs, kwargs):
        pool_size = self.pool_size
        if pool_size is None:
            pool_size = [2, 2]

        return tf.nn.max_pool_with_argmax(inputs, pool_size, strides=[1, 2, 2, 1], padding='SAME')


class MaxUnpooling2D(Layer):
    def init(self, pool_size=(2, 2), name=None):
        super(MaxUnpooling2D, self).init(name=name)
        self.pool_size = pool_size

    def call(self, inputs, kwargs, output_shape=None):
        pool_size = self.pool_size
        if pool_size is None:
            pool_size = [2, 2]

        return tf.nn.max_unpool(inputs, inputs, output_shape, ksize=pool_size, strides=[1, 2, 2, 1], padding='SAME')