import tensorflow as tf


class SimpleRNN(tf.keras.Model):
    def __init__(self, input_dim, output_dim, input_length, output_units, input_shapes):
        super(SimpleRNN, self).__init__()
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
        self.rnn_ = tf.keras.layers.SimpleRNN(units=self.output_dim, activation='tanh', name='rnn')
        self.output_ = tf.keras.layers.Dense(self.output_units, activation='softmax', name='output_layer')
        self.out = self.call(self.input_)

    def call(self, x):
        # 网络结构
        e = self.embedding_(x)
        r = self.rnn_(e)
        return self.output_(r)


class LSTM(tf.keras.Model):
    def __init__(self, input_dim, output_dim, input_length, output_units, input_shapes):
        super(LSTM, self).__init__()
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
        self.lstm_ = tf.keras.layers.LSTM(units=256, recurrent_activation='sigmoid', dropout=0.1, recurrent_dropout=0.1,
                                          name='lstm')
        self.output_ = tf.keras.layers.Dense(self.output_units, activation='softmax', name='output_layer')
        self.out = self.call(self.input_)

    def call(self, x):
        # 网络结构
        e = self.embedding_(x)
        r = self.lstm_(e)
        return self.output_(r)


class BiLSTM(tf.keras.Model):
    def __init__(self, input_dim, output_dim, input_length, output_units, input_shapes):
        super(BiLSTM, self).__init__()
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
        self.bilstm1_ = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=256, recurrent_activation='sigmoid', dropout=0.1, recurrent_dropout=0.1,
                                 name='bi-lstm1', return_sequences=True)
        )
        self.bilstm2_ = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=256, recurrent_activation='sigmoid', dropout=0.1, recurrent_dropout=0.1,
                                 name='bi-lstm2')
        )
        self.output_ = tf.keras.layers.Dense(self.output_units, activation='softmax', name='output_layer')
        self.out = self.call(self.input_)

    def call(self, x):
        # 网络结构
        e = self.embedding_(x)
        r1 = self.bilstm1_(e)
        r2 = self.bilstm2_(r1)
        return self.output_(r2)


class GRU(tf.keras.Model):
    def __init__(self, input_dim, output_dim, input_length, output_units, input_shapes):
        super(GRU, self).__init__()
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
        self.gru_ = tf.keras.layers.GRU(units=256, recurrent_activation='sigmoid',
                                        dropout=0.1, recurrent_dropout=0.1, name='gru')
        self.output_ = tf.keras.layers.Dense(self.output_units, activation='softmax', name='output_layer')
        self.out = self.call(self.input_)

    def call(self, x):
        # 网络结构
        e = self.embedding_(x)
        r = self.gru_(e)
        return self.output_(r)


class BiGRU(tf.keras.Model):
    def __init__(self, input_dim, output_dim, input_length, output_units, input_shapes):
        super(BiGRU, self).__init__()
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
        self.bigru1_ = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(units=256, recurrent_activation='sigmoid', dropout=0.1, recurrent_dropout=0.1,
                                name='bi-gru1', return_sequences=True)
        )
        self.bigru2_ = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(units=256, recurrent_activation='sigmoid', dropout=0.1, recurrent_dropout=0.1,
                                name='bi-gru2')
        )
        self.output_ = tf.keras.layers.Dense(self.output_units, activation='softmax', name='output_layer')
        self.out = self.call(self.input_)

    def call(self, x):
        # 网络结构
        e = self.embedding_(x)
        r1 = self.bigru1_(e)
        r2 = self.bigru2_(r1)
        return self.output_(r2)
