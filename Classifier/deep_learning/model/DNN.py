import tensorflow as tf
from ..define_layer import HierarchicalPooling


class DNN(tf.keras.Model):
    def __init__(self, output_units, input_shapes):
        super(DNN, self).__init__()
        # 参数
        self.output_units = output_units
        self.input_shapes = input_shapes

        # 层类型
        self.input_ = tf.keras.Input(shape=self.input_shapes, dtype='float64')
        self.dense1_ = tf.keras.layers.Dense(512, activation='relu')
        self.dense2_ = tf.keras.layers.Dense(256, activation='relu')
        self.dense3_ = tf.keras.layers.Dense(128, activation='relu')
        self.dense4_ = tf.keras.layers.Dense(64, activation='relu')
        self.output_ = tf.keras.layers.Dense(self.output_units, activation='softmax',
                                             kernel_regularizer=tf.keras.regularizers.l2(l2=0.01))
        self.out = self.call(self.input_)

    def call(self, x):
        # 网络结构
        x = self.dense1_(x)
        x = self.dense2_(x)
        x = self.dense3_(x)
        x = self.dense4_(x)
        return self.output_(x)


class fastText(tf.keras.Model):
    def __init__(self, input_dim, output_dim, input_length, output_units, input_shapes):
        super(fastText, self).__init__()
        # 参数
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.output_units = output_units
        self.input_shapes = input_shapes

        # 层类型
        self.input_ = tf.keras.Input(shape=self.input_shapes, dtype='float64')
        self.embedding_ = tf.keras.layers.Embedding(input_dim=self.input_dim, output_dim=self.output_dim,
                                                    input_length=self.input_length, name='embedding')
        self.pooling_ = tf.keras.layers.GlobalAveragePooling1D(name='global_average')
        self.output_ = tf.keras.layers.Dense(self.output_units, activation='softmax',
                                             kernel_regularizer=tf.keras.regularizers.l2(), name='output_layer')
        self.out = self.call(self.input_)

    def call(self, x):
        # 网络结构
        x = self.embedding_(x)
        x = self.pooling_(x)
        return self.output_(x)


class DAN(tf.keras.Model):
    def __init__(self, input_dim, output_dim, input_length, output_units, input_shapes):
        super(DAN, self).__init__()
        # 参数
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.output_units = output_units
        self.input_shapes = input_shapes

        # 层类型
        self.input_ = tf.keras.Input(shape=self.input_shapes, dtype='float64')
        self.embedding_ = tf.keras.layers.Embedding(input_dim=self.input_dim, output_dim=self.output_dim,
                                                    input_length=self.input_length, name='embedding')
        self.dropout_ = tf.keras.layers.Dropout(rate=0.05, name='word_dropout')
        self.pooling_ = tf.keras.layers.GlobalAveragePooling1D(name='global_average')
        self.dense1_ = tf.keras.layers.Dense(100, activation='relu', name='hidden_layer1')
        self.dense2_ = tf.keras.layers.Dense(100, activation='relu', name='hidden_layer2')
        self.output_ = tf.keras.layers.Dense(self.output_units, activation='softmax',
                                             kernel_regularizer=tf.keras.regularizers.l2(l2=0.01), name='output_layer')
        self.out = self.call(self.input_)

    def call(self, x):
        # 网络结构
        e = self.embedding_(x)
        e1 = self.dropout_(e)
        p = self.pooling_(e1)
        d1 = self.dense1_(p)
        d2 = self.dense2_(d1)
        return self.output_(d2)


class SWEM(tf.keras.Model):
    def __init__(self, input_dim, output_dim, input_length, output_units, input_shapes,
                 pooling_type='hierarchical', kernel_size=3):
        super(SWEM, self).__init__()
        # 参数
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.output_units = output_units
        self.pooling_type = pooling_type
        self.kernel_size = kernel_size
        self.input_shapes = input_shapes

        self.input_ = tf.keras.Input(shape=self.input_shapes, dtype='float64')
        self.embedding_ = tf.keras.layers.Embedding(input_dim=self.input_dim, output_dim=self.output_dim,
                                                    input_length=self.input_length)
        self.maxpool_ = tf.keras.layers.GlobalMaxPooling1D()
        self.avgpool_ = tf.keras.layers.GlobalAveragePooling1D()
        self.hierarchicalpool_ = HierarchicalPooling(self.input_length, self.kernel_size)
        self.output_ = tf.keras.layers.Dense(self.output_units, activation='softmax')
        self.out = self.call(self.input_)

    def call(self, x):
        e = self.embedding_(x)
        if self.pooling_type == 'hierarchical':
            p = self.hierarchicalpool_(e)
        elif self.pooling_type == 'max':
            p = self.maxpool_(e)
        elif self.pooling_type == 'avg':
            p = self.avgpool_(e)
        elif self.pooling_type == 'concat':
            p1 = self.maxpool_(e)
            p2 = self.avgpool_(e)
            p = tf.keras.layers.concatenate([p1, p2])
        else:
            raise RuntimeError("encode_type must be 'max', 'avg', 'concat', 'hierarchical'")
        o = self.output_(p)
        return o
