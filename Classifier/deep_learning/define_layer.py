import tensorflow as tf


# 宽卷积层
class WideConvolution1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        super(WideConvolution1D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(WideConvolution1D, self).build(input_shape)

    def call(self, x):
        x_pad = tf.keras.layers.ZeroPadding1D((self.kernel_size - 1, self.kernel_size - 1))(x)
        x_conv = tf.keras.layers.Convolution1D(filters=self.filters, kernel_size=self.kernel_size, strides=1,
                                               padding='valid', kernel_initializer='normal', activation='tanh')(x_pad)
        return x_conv

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1] + self.kernel_size - 1, input_shape[-1]


# K最大池化层
class KMaxPooling(tf.keras.layers.Layer):
    def __init__(self, topk, **kwargs):
        self.topk = topk
        super(KMaxPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        super(KMaxPooling, self).build(input_shape)

    def call(self, x):
        x_reshape = tf.transpose(x, perm=[0, 2, 1])
        x_pool = tf.math.top_k(input=x_reshape, k=self.topk, sorted=False).values
        x_pool_reshape = tf.transpose(x_pool, perm=[0, 2, 1])
        return x_pool_reshape

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.top_k, input_shape[-1]


# 折叠层
class Folding(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Folding, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Folding, self).build(input_shape)

    def call(self, x):
        x_conv1 = x[:, :, ::2]
        x_conv2 = x[:, :, 1::2]
        x_fold = tf.keras.layers.Add()([x_conv1, x_conv2])
        return x_fold

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], int(input_shape[2] / 2)


# 层次池化层
class HierarchicalPooling(tf.keras.layers.Layer):
    def __init__(self, input_length, kernel_size,  **kwargs):
        super(HierarchicalPooling, self).__init__(**kwargs)
        self.input_length = input_length
        self.kernel_size = kernel_size

    def build(self, input_shape):
        super(HierarchicalPooling, self).build(input_shape)

    def call(self, x):
        pools = []
        for i in range(self.input_length - self.kernel_size + 1):
            x_mean = tf.reduce_mean(x[:, i:i+self.kernel_size, :], axis=1)
            x_dim = tf.expand_dims(x_mean, axis=-1)
            pools.append(x_dim)
        pools = tf.concat(pools, axis=-1)
        x_max = tf.reduce_max(pools, axis=-1)
        return x_max

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


# 注意力层
class Attention(tf.keras.layers.Layer):
    def __init__(self, attention_dim=None, **kwargs):
        self.attention_dim = attention_dim
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.attention_dim is None:
            attention_dim = input_shape[-1]
        else:
            attention_dim = self.attention_dim

        self.W = self.add_weight(name='W',
                                 shape=(input_shape[-1], attention_dim),
                                 initializer='glorot_normal',
                                 trainable=True)
        self.b = self.add_weight(name='b',
                                 shape=(attention_dim,),
                                 initializer='zeros',
                                 trainable=True)
        self.u = self.add_weight(name='u',
                                 shape=(attention_dim, 1),
                                 initializer='glorot_normal',
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        wt = tf.keras.backend.dot(x, self.W)  # wt = W * x
        bt = tf.keras.backend.tanh(tf.keras.backend.bias_add(wt, self.b))  # bt = Wt + b
        ut = tf.keras.backend.dot(bt, self.u)  # ut = bt * u
        att = tf.keras.backend.exp(ut)  # att = exp(ut)
        at = att / tf.keras.backend.sum(att, axis=1, keepdims=True)  # 按行相加，并且保持其二维特性
        if mask is not None:
            at *= tf.keras.backend.cast(mask, tf.keras.backend.floatx())
        st = x * at
        s = tf.keras.backend.sum(st, axis=1)
        return s

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        return (input_shape[0], output_len)


# 自注意力层
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, attention_dim, **kwargs):
        self.attention_dim = attention_dim
        super(SelfAttention, self).__init__(self)

    def build(self, input_shape):
        self.W = self.add_weight(name='QKV', shape=(3, input_shape[2], self.attention_dim),
                                 initializer='uniform', regularizer=tf.keras.regularizers.L1L2(l1=0.0000032, l2=0),
                                 trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        WQ = tf.keras.backend.dot(x, self.W[0])
        WK = tf.keras.backend.dot(x, self.W[1])
        WV = tf.keras.backend.dot(x, self.W[2])
        QK = tf.keras.backend.batch_dot(WQ, tf.transpose(WK, perm=[0, 2, 1]))
        QK = QK / (self.attention_dim ** 0.5)
        QK = tf.keras.backend.softmax(QK)
        output = tf.keras.backend.batch_dot(QK, WV)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.attention_dim)


# 多头自注意力层
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, head_num, attention_dim):
        self.head_num = head_num
        self.attention_dim = attention_dim
        super(MultiHeadSelfAttention, self).__init__(self)

    def build(self, input_shape):
        self.W = self.add_weight(name='QKV', shape=(self.head_num, 3, input_shape[2], self.attention_dim),
                                 initializer='uniform', regularizer=tf.keras.regularizers.L1L2(l1=0.0000032, l2=0),
                                 trainable=True)
        self.Wz = self.add_weight(name='Z', shape=(self.head_num * self.attention_dim, self.attention_dim),
                                  initializer='uniform', regularizer=tf.keras.regularizers.L1L2(l1=0.0000032, l2=0),
                                  trainable=True)
        super(MultiHeadSelfAttention, self).build(input_shape)

    def call(self, x):
        WQ = tf.keras.backend.dot(x, self.W[0, 0])
        WK = tf.keras.backend.dot(x, self.W[0, 1])
        WV = tf.keras.backend.dot(x, self.W[0, 2])
        QK = tf.keras.backend.batch_dot(WQ, tf.transpose(WK, perm=[0, 2, 1]))
        QK = QK / (self.attention_dim ** 0.5)
        QK = tf.keras.backend.softmax(QK)
        outputs = tf.keras.backend.batch_dot(QK, WV)
        for num in range(1, self.head_num):
            WQ = tf.keras.backend.dot(x, self.W[num, 0])
            WK = tf.keras.backend.dot(x, self.W[num, 1])
            WV = tf.keras.backend.dot(x, self.W[num, 2])
            QK = tf.keras.backend.batch_dot(WQ, tf.transpose(WK, perm=[0, 2, 1]))
            QK = QK / (self.attention_dim ** 0.5)
            QK = tf.keras.backend.softmax(QK)
            output = tf.keras.backend.batch_dot(QK, WV)
            outputs = tf.keras.backend.concatenate([outputs, output])
        Z = tf.keras.backend.dot(outputs, self.Wz)
        return Z

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.attention_dim)


