import tensorflow as tf
from ..define_layer import WideConvolution1D
from ..define_layer import KMaxPooling
from ..define_layer import Folding


class LeNet_5(tf.keras.Model):
    def __init__(self, input_dim, input_length, output_units, input_shapes):
        super(LeNet_5, self).__init__()
        # 参数
        self.input_dim = input_dim
        self.input_length = input_length
        self.output_units = output_units
        self.input_shapes = input_shapes

        # 层类型
        self.input_ = tf.keras.Input(shape=self.input_shapes, dtype='float64')
        self.embedding_ = tf.keras.layers.Embedding(input_dim=self.input_dim, output_dim=100,
                                                    input_length=self.input_length, name='embedding')
        self.conv1d1_ = tf.keras.layers.Convolution1D(filters=256, kernel_size=3, strides=1, padding='same',
                                                      name='conv1D_1')
        self.maxpool1_ = tf.keras.layers.MaxPooling1D(pool_size=3, strides=3, padding='same', name='maxpooling1')
        self.conv1d2_ = tf.keras.layers.Convolution1D(filters=128, kernel_size=3, strides=1, padding='same',
                                                      name='conv1D_2')
        self.maxpool2_ = tf.keras.layers.MaxPooling1D(pool_size=3, strides=3, padding='same', name='maxpooling2')
        self.conv1d3_ = tf.keras.layers.Convolution1D(filters=64, kernel_size=3, strides=1, padding='same',
                                                      name='conv1D_3')
        self.flatten_ = tf.keras.layers.Flatten(name='flatten')
        self.dropout1_ = tf.keras.layers.Dropout(rate=0.1, name='dropout1')
        self.bn_ = tf.keras.layers.BatchNormalization(name='BN')
        self.dense_ = tf.keras.layers.Dense(units=256, activation='relu', name='hiden_layer')
        self.dropout2_ = tf.keras.layers.Dropout(rate=0.1, name='dropout2')
        self.output_ = tf.keras.layers.Dense(self.output_units, activation='softmax', name='output_layer')
        self.out = self.call(self.input_)

    def call(self, x):
        # 网络结构
        e = self.embedding_(x)
        c1 = self.conv1d1_(e)
        p1 = self.maxpool1_(c1)
        c2 = self.conv1d2_(p1)
        p2 = self.maxpool2_(c2)
        c3 = self.conv1d3_(p2)
        f = self.flatten_(c3)
        fdrop = self.dropout1_(f)
        bn = self.bn_(fdrop)
        d = self.dense_(bn)
        ddrop = self.dropout2_(d)
        return self.output_(ddrop)


class TextCNN(tf.keras.Model):
    def __init__(self, input_dim, output_dim, input_length, embedding_weight, output_units, input_shapes):
        super(TextCNN, self).__init__()
        # 参数
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.embedding_weight = embedding_weight
        self.output_units = output_units
        self.input_shapes = input_shapes

        # 层类型
        self.input_ = tf.keras.Input(shape=self.input_shapes, dtype='float64')
        self.embedding_ = tf.keras.layers.Embedding(input_dim=self.input_dim, output_dim=self.output_dim,
                                                    input_length=self.input_length,
                                                    weights=[self.embedding_weight],
                                                    trainable=True, name='pre_training_embedding')
        self.conv1d1_ = tf.keras.layers.Convolution1D(filters=256, kernel_size=3, strides=1, padding='same',
                                                      name='conv1D_1')
        self.maxpool1_ = tf.keras.layers.MaxPooling1D(pool_size=4, name='maxpooling1')
        self.conv1d2_ = tf.keras.layers.Convolution1D(filters=256, kernel_size=4, strides=1, padding='same',
                                                      name='conv1D_2')
        self.maxpool2_ = tf.keras.layers.MaxPooling1D(pool_size=4, name='maxpooling2')
        self.conv1d3_ = tf.keras.layers.Convolution1D(filters=256, kernel_size=5, strides=1, padding='same',
                                                      name='conv1D_3')
        self.maxpool3_ = tf.keras.layers.MaxPooling1D(pool_size=4, name='maxpooling3')

        self.flatten_ = tf.keras.layers.Flatten(name='flatten')
        self.dropout_ = tf.keras.layers.Dropout(rate=0.2, name='dropout')
        self.output_ = tf.keras.layers.Dense(self.output_units, activation='softmax', name='output_layer')
        self.out = self.call(self.input_)

    def call(self, x):
        # 网络结构
        e = self.embedding_(x)
        c1 = self.conv1d1_(e)
        p1 = self.maxpool1_(c1)
        c2 = self.conv1d2_(e)
        p2 = self.maxpool2_(c2)
        c3 = self.conv1d3_(e)
        p3 = self.maxpool3_(c3)
        con = tf.keras.layers.concatenate([p1, p2, p3], axis=-1)

        f = self.flatten_(con)
        fdrop = self.dropout_(f)
        return self.output_(fdrop)


class TextDCNNBlock(tf.keras.Model):
    def __init__(self, channel, filter_list, input_length):
        super(TextDCNNBlock, self).__init__()
        # 参数
        self.channel = channel
        self.filter_list = filter_list
        self.input_length = input_length

        self.conv_pool = tf.keras.models.Sequential()
        for i in range(len(filter_list) - 1):
            k = max(3, int(((len(filter_list) - (i + 1)) / len(filter_list)) * self.input_length))  # k取值见论文
            # 层结构
            self.wide_conv = WideConvolution1D(filters=256, kernel_size=self.filter_list[i])
            self.conv_pool.add(self.wide_conv)
            self.kmax_pool = KMaxPooling(topk=k)
            self.conv_pool.add(self.kmax_pool)
        k = max(3, int(((len(filter_list) - len(filter_list)) / len(filter_list)) * self.input_length))
        self.wide_conv1 = WideConvolution1D(filters=256, kernel_size=self.filter_list[-1])
        self.fold1 = Folding()
        self.kmax_pool1 = KMaxPooling(topk=k)

    def call(self, x):
        cp = self.conv_pool(x)
        c = self.wide_conv1(cp)
        f = self.fold1(c)
        kp = self.kmax_pool1(f)
        return kp


class TextDCNN(tf.keras.Model):
    def __init__(self, input_dim, output_dim, input_length, output_units, filters_list, input_shapes, **kwargs):
        super(TextDCNN, self).__init__(**kwargs)
        # 参数
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.output_units = output_units
        self.filters_list = filters_list
        self.input_shapes = input_shapes

        # 层类型
        self.input_ = tf.keras.Input(shape=self.input_shapes, dtype='float64')
        self.embedding_ = tf.keras.layers.Embedding(input_dim=self.input_dim, output_dim=self.output_dim,
                                                    input_length=self.input_length)
        self.dropout_ = tf.keras.layers.Dropout(rate=0.2)
        self.flatten_ = tf.keras.layers.Flatten()
        self.output_ = tf.keras.layers.Dense(self.output_units, activation='softmax')
        self.out = self.call(self.input_)

    def call(self, x):
        e = self.embedding_(x)
        pools = []
        for channel, filter_list in enumerate(self.filters_list):
            b = TextDCNNBlock(channel, filter_list, self.input_length)(e)
            pools.append(b)
        con = tf.keras.layers.concatenate(pools, axis=1)
        drop = self.dropout_(con)
        f = self.flatten_(drop)
        o = self.output_(f)
        return o


class DPCNNBlock(tf.keras.Model):
    def __init__(self, block_id):
        super(DPCNNBlock, self).__init__()
        self.block_id = block_id
        self.conv1d1_ = tf.keras.layers.Convolution1D(filters=256, kernel_size=3, strides=1, padding='same',
                                                      activation='linear',
                                                      kernel_regularizer=tf.keras.regularizers.l2(0.00001),
                                                      bias_regularizer=tf.keras.regularizers.l2(0.00001))
        self.bn1_ = tf.keras.layers.BatchNormalization()
        self.activation1_ = tf.keras.layers.PReLU()
        self.conv1d2_ = tf.keras.layers.Convolution1D(filters=256, kernel_size=3, strides=1, padding='same',
                                                      activation='linear',
                                                      kernel_regularizer=tf.keras.regularizers.l2(0.00001),
                                                      bias_regularizer=tf.keras.regularizers.l2(0.00001))
        self.bn2_ = tf.keras.layers.BatchNormalization()
        self.activation2_ = tf.keras.layers.PReLU()
        self.pool_ = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2)

    def call(self, x):
        c1 = self.conv1d1_(x)
        b1 = self.bn1_(c1)
        a1 = self.activation1_(b1)
        c2 = self.conv1d2_(a1)
        b2 = self.bn2_(c2)
        a2 = self.activation2_(b2)
        add = tf.keras.layers.add([a2, x])
        if self.block_id != 6:
            return self.pool_(add)
        else:
            return add


class DPCNN(tf.keras.Model):
    def __init__(self, blocks, input_dim, output_dim, input_length, embedding_weight, output_units, input_shapes):
        super(DPCNN, self).__init__()
        # 参数
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.embedding_weight = embedding_weight
        self.output_units = output_units
        self.input_shapes = input_shapes
        # 层类型
        self.input_ = tf.keras.Input(shape=self.input_shapes, dtype='float64')
        self.embedding_ = tf.keras.layers.Embedding(input_dim=self.input_dim, output_dim=self.output_dim,
                                                    input_length=self.input_length,
                                                    weights=[self.embedding_weight], trainable=False)
        self.dropout1_ = tf.keras.layers.SpatialDropout1D(0.2)
        self.conv1d1_ = tf.keras.layers.Convolution1D(filters=256, kernel_size=1, strides=1, padding='same',
                                                      activation='linear',
                                                      kernel_regularizer=tf.keras.regularizers.l2(0.00001),
                                                      bias_regularizer=tf.keras.regularizers.l2(0.00001))
        self.activation1_ = tf.keras.layers.PReLU()

        self.blocks_ = tf.keras.models.Sequential()
        for block_id in range(blocks):
            block = DPCNNBlock(block_id)
            self.blocks_.add(block)
        self.global_pool_ = tf.keras.layers.GlobalMaxPooling1D()

        self.dense_ = tf.keras.layers.Dense(256, activation='linear')
        self.bn_ = tf.keras.layers.BatchNormalization()
        self.activation2_ = tf.keras.layers.PReLU()
        self.dropout2_ = tf.keras.layers.Dropout(0.5)
        self.output_ = tf.keras.layers.Dense(self.output_units, activation='softmax')
        self.out = self.call(self.input_)

    def call(self, x):
        e = self.embedding_(x)
        drop1 = self.dropout1_(e)
        c1 = self.conv1d1_(drop1)
        a1 = self.activation1_(c1)
        bl = self.blocks_(a1)
        gp = self.global_pool_(bl)
        dense = self.dense_(gp)
        bn = self.bn_(dense)
        a2 = self.activation2_(bn)
        drop2 = self.dropout2_(a2)
        return self.output_(drop2)


class TextInception(tf.keras.Model):
    def __init__(self, input_dim, output_dim, input_length, output_units, input_shapes):
        super(TextInception, self).__init__()
        # 参数
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.output_units = output_units
        self.input_shapes = input_shapes

        # 层类型
        self.input_ = tf.keras.Input(shape=self.input_shapes, dtype='float64')
        self.embedding_ = tf.keras.layers.Embedding(input_dim=self.input_dim, output_dim=self.output_dim,
                                                    input_length=self.input_length)
        self.conv_block1_ = tf.keras.layers.Convolution1D(filters=128, kernel_size=1, strides=1, padding='same')
        self.conv_block2_ = tf.keras.models.Sequential(
            [tf.keras.layers.Convolution1D(filters=256, kernel_size=1, strides=1, padding='same'),
             tf.keras.layers.BatchNormalization(),
             tf.keras.layers.ReLU(),
             tf.keras.layers.Convolution1D(filters=128, kernel_size=3, strides=1, padding='same')])
        self.conv_block3_ = tf.keras.models.Sequential(
            [tf.keras.layers.Convolution1D(filters=256, kernel_size=3, strides=1, padding='same'),
             tf.keras.layers.BatchNormalization(),
             tf.keras.layers.ReLU(),
             tf.keras.layers.Convolution1D(filters=128, kernel_size=5, strides=1, padding='same')])
        self.conv_block4_ = tf.keras.layers.Convolution1D(filters=128, kernel_size=3, strides=1, padding='same')
        self.dense_block_ = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                                        tf.keras.layers.Dense(128),
                                                        tf.keras.layers.Dropout(rate=0.2),
                                                        tf.keras.layers.BatchNormalization(),
                                                        tf.keras.layers.ReLU()])
        self.output_ = tf.keras.layers.Dense(self.output_units, activation='softmax')
        self.out = self.call(self.input_)

    def call(self, x):
        e = self.embedding_(x)
        b1 = self.conv_block1_(e)
        b2 = self.conv_block2_(e)
        b3 = self.conv_block3_(e)
        b4 = self.conv_block4_(e)
        con = tf.keras.layers.concatenate([b1, b2, b3, b4], axis=-1)
        d = self.dense_block_(con)
        return self.output_(d)


class TextVDCNNBlock(tf.keras.Model):
    def __init__(self, filters, end=False):
        super(TextVDCNNBlock, self).__init__()
        self.filters = filters
        self.end = end

        self.conv1_ = tf.keras.layers.Convolution1D(filters=self.filters, kernel_size=3, strides=1, padding='same',
                                                    kernel_regularizer=tf.keras.regularizers.l2(0.0000032),
                                                    bias_regularizer=tf.keras.regularizers.l2(0.0000032),
                                                    activation='linear')
        self.bn1_ = tf.keras.layers.BatchNormalization()
        self.relu1_ = tf.keras.layers.ReLU()
        self.conv2_ = tf.keras.layers.Convolution1D(filters=self.filters, kernel_size=3, strides=1, padding='same',
                                                    kernel_regularizer=tf.keras.regularizers.l2(0.0000032),
                                                    bias_regularizer=tf.keras.regularizers.l2(0.0000032),
                                                    activation='linear')
        self.bn2_ = tf.keras.layers.BatchNormalization()
        self.relu2_ = tf.keras.layers.ReLU()
        self.conv3_ = tf.keras.layers.Convolution1D(filters=self.filters, kernel_size=1, strides=2, padding='same')
        self.bn3_ = tf.keras.layers.BatchNormalization()
        self.pool_ = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same')

    def call(self, x):
        c1 = self.conv1_(x)
        bn1 = self.bn1_(c1)
        r1 = self.relu1_(bn1)
        c2 = self.conv2_(r1)
        bn2 = self.bn2_(c2)
        r2 = self.relu2_(bn2)
        if self.end:
            c3 = self.conv3_(x)
            bn3 = self.bn3_(c3)
            p = self.pool_(r2)
            o = tf.keras.layers.add([p, bn3])
        else:
            o = tf.keras.layers.add([r2, x])
        return o


class TextVDCNN(tf.keras.Model):
    def __init__(self, input_dim, output_dim, input_length, output_units, input_shapes,
                 init_filters=64, blocks=[2, 2, 2, 2]):
        super(TextVDCNN, self).__init__()
        # 参数
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.output_units = output_units
        self.init_filters = init_filters
        self.blocks = blocks
        self.end = False
        self.input_shapes = input_shapes

        # 层类型
        self.input_ = tf.keras.Input(shape=self.input_shapes, dtype='float64')
        self.embedding_ = tf.keras.layers.Embedding(input_dim=self.input_dim, output_dim=self.output_dim,
                                                    input_length=self.input_length)
        self.conv_ = tf.keras.layers.Convolution1D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=True,
                                                   kernel_initializer='he_normal')
        self.blocks_net_ = tf.keras.models.Sequential()
        for num, b in enumerate(blocks):
            if num != 0:
                self.init_filters = self.init_filters*2
            for i in range(b):
                if i == b - 1:
                    self.end = True
                self.blocks_net_.add(TextVDCNNBlock(filters=self.init_filters, end=self.end))
        self.k_pool_ = KMaxPooling(topk=2)
        self.flat_ = tf.keras.layers.Flatten()
        self.dense1_ = tf.keras.layers.Dense(2048, activation='relu')
        self.dense2_ = tf.keras.layers.Dense(2048, activation='relu')
        self.output_ = tf.keras.layers.Dense(self.output_units, activation='softmax')
        self.out = self.call(self.input_)

    def call(self, x):
        e = self.embedding_(x)
        conv = self.conv_(e)
        b = self.blocks_net_(conv)
        kp = self.k_pool_(b)
        f = self.flat_(kp)
        d1 = self.dense1_(f)
        d2 = self.dense2_(d1)
        o = self.output_(d2)
        return o










