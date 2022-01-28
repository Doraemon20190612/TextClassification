import tensorflow as tf
from ..define_layer import Attention, SelfAttention


class BiLSTMAttention(tf.keras.Model):
    def __init__(self, input_dim, output_dim, input_length, output_units, input_shapes):
        super(BiLSTMAttention, self).__init__()
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
        self.bilstm_ = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=256, recurrent_activation='sigmoid', dropout=0.1, recurrent_dropout=0.1,
                                 return_sequences=True, name='bi-lstm')
        )
        self.attention_ = Attention()
        self.output_ = tf.keras.layers.Dense(self.output_units, activation='softmax', name='output_layer')
        self.out = self.call(self.input_)

    def call(self, x):
        # 网络结构
        e = self.embedding_(x)
        r = self.bilstm_(e)
        a = self.attention_(r)
        return self.output_(a)


class SelfAttentionModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim, input_length, output_units, input_shapes):
        super(SelfAttentionModel, self).__init__()
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
        self.self_attention_ = SelfAttention(self.output_dim)
        self.global_ = tf.keras.layers.GlobalAveragePooling1D()
        self.dropout_ = tf.keras.layers.Dropout(0.5)
        self.output_ = tf.keras.layers.Dense(self.output_units, activation='softmax', name='output_layer')
        self.out = self.call(self.input_)

    def call(self, x):
        e = self.embedding_(x)
        sa = self.self_attention_(e)
        g = self.global_(sa)
        d = self.dropout_(g)
        output = self.output_(d)
        return output


class BiLSTMSelfAttention(tf.keras.Model):
    def __init__(self, input_dim, output_dim, input_length, output_units, input_shapes):
        super(BiLSTMSelfAttention, self).__init__()
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
        self.self_attention_ = SelfAttention(attention_dim=self.output_dim)
        self.bilstm_ = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=256, recurrent_activation='sigmoid', dropout=0.1, recurrent_dropout=0.1,
                                 name='bi-lstm')
        )
        self.output_ = tf.keras.layers.Dense(self.output_units, activation='softmax', name='output_layer')
        self.out = self.call(self.input_)

    def call(self, x):
        e = self.embedding_(x)
        sa = self.self_attention_(e)
        bl = self.bilstm_(sa)
        output = self.output_(bl)
        return output


