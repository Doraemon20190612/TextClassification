import tensorflow as tf


class RCNN(tf.keras.Model):
    def __init__(self, input_dim, output_dim, input_length, output_units):
        super(RCNN, self).__init__()
        # 参数
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.output_units = output_units

        # 层类型
        self.embedding_ = tf.keras.layers.Embedding(input_dim=self.input_dim, output_dim=self.output_dim,
                                                    input_length=self.input_length, name='embedding_center')
        self.lstm_forward_ = tf.keras.layers.LSTM(units=256, recurrent_activation='sigmoid', dropout=0.1,
                                                  recurrent_dropout=0.1,
                                                  return_sequences=True, name='lstm_forward')
        self.lstm_backward_ = tf.keras.layers.LSTM(units=256, recurrent_activation='sigmoid', dropout=0.1,
                                                   recurrent_dropout=0.1,
                                                   return_sequences=True, go_backwards=True, name='lstm_backward')
        self.dense_ = tf.keras.layers.Dense(units=128, activation='tanh', name='time_dense')
        self.poolling_ = tf.keras.layers.GlobalMaxPooling1D(name='poolling')
        self.output_ = tf.keras.layers.Dense(self.output_units, activation='softmax', name='output_layer')

    def call(self, x):
        # 网络结构
        e = self.embedding_(x)
        f = self.lstm_forward_(e)
        b = self.lstm_backward_(e)
        t = tf.keras.layers.Concatenate(axis=2)([f, e, b])
        d = tf.keras.layers.TimeDistributed(self.dense_)(t)
        p = self.poolling_(d)
        return self.output_(p)


class CLSTM(tf.keras.Model):
    def __init__(self, input_dim, output_dim, input_length, output_units, input_shapes):
        super(CLSTM, self).__init__()
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
        self.conv1d_ = tf.keras.layers.Convolution1D(filters=256, kernel_size=3, strides=1, padding='same',
                                                     activation='relu', name='conv1D')
        self.maxpool_ = tf.keras.layers.MaxPooling1D(pool_size=4, name='maxpooling')
        self.lstm1_ = tf.keras.layers.LSTM(units=256, recurrent_activation='sigmoid', dropout=0.1,
                                           recurrent_dropout=0.1, return_sequences=True, name='lstm1')
        self.lstm2_ = tf.keras.layers.LSTM(units=256, recurrent_activation='sigmoid', dropout=0.1,
                                           recurrent_dropout=0.1, name='lstm2')
        self.output_ = tf.keras.layers.Dense(self.output_units, activation='softmax', name='output_layer')
        self.out = self.call(self.input_)

    def call(self, x):
        # 网络结构
        e = self.embedding_(x)
        c = self.conv1d_(e)
        p = self.maxpool_(c)
        r1 = self.lstm1_(p)
        r2 = self.lstm2_(r1)
        return self.output_(r2)


class paraCLSTM(tf.keras.Model):
    def __init__(self, input_dim, output_dim, input_length, output_units, input_shapes):
        super(paraCLSTM, self).__init__()
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
        self.conv1d_ = tf.keras.layers.Convolution1D(filters=256, kernel_size=3, strides=1, padding='same',
                                                     activation='relu', name='conv1D')
        self.maxpool_ = tf.keras.layers.MaxPooling1D(pool_size=4, name='maxpooling')
        self.flatten_ = tf.keras.layers.Flatten(name='flatten')
        self.dense1_ = tf.keras.layers.Dense(units=256, name='dense1')
        self.bilstm_ = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=256, recurrent_activation='sigmoid', dropout=0.1, recurrent_dropout=0.1,
                                 name='bi-lstm')
        )
        self.dense2_ = tf.keras.layers.Dense(units=256, name='dense2')
        self.output_ = tf.keras.layers.Dense(self.output_units, activation='softmax', name='output_layer')
        self.out = self.call(self.input_)

    def call(self, x):
        # 网络结构
        e = self.embedding_(x)
        c = self.conv1d_(e)
        p = self.maxpool_(c)
        f = self.flatten_(p)
        d1 = self.dense1_(f)
        r = self.bilstm_(e)
        d2 = self.dense2_(r)
        con = tf.keras.layers.Concatenate(axis=-1)([d1, d2])
        return self.output_(con)
