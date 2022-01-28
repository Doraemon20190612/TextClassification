import tensorflow as tf
from sklearn import neural_network
import numpy as np
from .model.DNN import DNN, fastText, DAN, SWEM
from .model.CNN import LeNet_5, TextCNN, TextDCNN, TextInception, DPCNN, TextVDCNN
from .model.RNN import SimpleRNN, LSTM, BiLSTM, GRU, BiGRU
from .model.MixNN import RCNN, CLSTM, paraCLSTM
from .model.Attention import BiLSTMAttention, SelfAttentionModel, BiLSTMSelfAttention


def mlp(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    ###########################################

    clf_model = neural_network.MLPClassifier()
    clf_model.fit(x_train, y_train)

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def dnn(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    x_valid = input_['x_valid']
    y_valid = input_['y_valid']
    parameter = input_['parameter']
    ###########################################
    print(x_train.shape)
    clf_model = DNN(output_units=len(set(y_train)),
                    input_shapes=(x_train.shape[1],))

    # 模型配置
    clf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
    clf_model.fit(x_train, y_train,
                  batch_size=parameter['Classifier']['deep_learning']['public']['batch_size'],
                  epochs=parameter['Classifier']['deep_learning']['public']['epochs'],
                  validation_data=(x_valid, y_valid))
    clf_model.summary()

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def fast_text(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    x_valid = input_['x_valid']
    y_valid = input_['y_valid']
    word_indexs = input_['word_indexs']
    parameter = input_['parameter']
    ###########################################
    doc_maxlen = x_train.shape[1]

    clf_model = fastText(input_dim=word_indexs,
                         output_dim=100,
                         input_length=doc_maxlen,
                         output_units=len(set(y_train)),
                         input_shapes=(doc_maxlen,))
    # 模型配置
    clf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
    clf_model.fit(x_train, y_train,
                  batch_size=parameter['Classifier']['deep_learning']['public']['batch_size'],
                  epochs=parameter['Classifier']['deep_learning']['public']['epochs'],
                  validation_data=(x_valid, y_valid))
    clf_model.summary()

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def dan(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    x_valid = input_['x_valid']
    y_valid = input_['y_valid']
    word_indexs = input_['word_indexs']
    parameter = input_['parameter']
    ###########################################

    doc_maxlen = x_train.shape[1]

    clf_model = DAN(input_dim=word_indexs,
                    output_dim=100,
                    input_length=doc_maxlen,
                    output_units=len(set(y_train)),
                    input_shapes=(doc_maxlen,))

    # 模型配置
    clf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
    clf_model.fit(x_train, y_train,
                  batch_size=parameter['Classifier']['deep_learning']['public']['batch_size'],
                  epochs=parameter['Classifier']['deep_learning']['public']['epochs'],
                  validation_data=(x_valid, y_valid))
    clf_model.summary()

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def swem(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    x_valid = input_['x_valid']
    y_valid = input_['y_valid']
    word_indexs = input_['word_indexs']
    parameter = input_['parameter']
    ###########################################

    doc_maxlen = x_train.shape[1]
    clf_model = SWEM(input_dim=word_indexs, output_dim=100, input_length=doc_maxlen,
                     output_units=len(set(y_train)),input_shapes=(doc_maxlen,),
                     pooling_type=parameter['Classifier']['deep_learning']['swem']['pooling_type'],
                     kernel_size=parameter['Classifier']['deep_learning']['swem']['kernel_size'])

    # 模型配置
    clf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
    clf_model.fit(x_train, y_train,
                  batch_size=parameter['Classifier']['deep_learning']['public']['batch_size'],
                  epochs=parameter['Classifier']['deep_learning']['public']['epochs'],
                  validation_data=(x_valid, y_valid))
    clf_model.summary()

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def lenet_5(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    x_valid = input_['x_valid']
    y_valid = input_['y_valid']
    word_indexs = input_['word_indexs']
    parameter = input_['parameter']
    ###########################################

    doc_maxlen = x_train.shape[1]
    clf_model = LeNet_5(input_dim=word_indexs,
                        input_length=doc_maxlen,
                        output_units=len(set(y_train)),
                        input_shapes=(doc_maxlen,))

    # 模型配置
    clf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
    clf_model.fit(x_train, y_train,
                  batch_size=parameter['Classifier']['deep_learning']['public']['batch_size'],
                  epochs=parameter['Classifier']['deep_learning']['public']['epochs'],
                  validation_data=(x_valid, y_valid))
    clf_model.summary()

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def text_cnn(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    x_valid = input_['x_valid']
    y_valid = input_['y_valid']
    word_indexs = input_['word_indexs']
    parameter = input_['parameter']
    wordvec_model = input_['wordvec_model']
    embedding_weight = input_['embedding_weight']
    ###########################################

    doc_maxlen = x_train.shape[1]
    clf_model = TextCNN(input_dim=word_indexs,
                        output_dim=wordvec_model.dim,
                        input_length=doc_maxlen,
                        embedding_weight=embedding_weight,
                        output_units=len(set(y_train)),
                        input_shapes=(doc_maxlen,))

    # 模型配置
    clf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
    clf_model.fit(x_train, y_train,
                  batch_size=parameter['Classifier']['deep_learning']['public']['batch_size'],
                  epochs=parameter['Classifier']['deep_learning']['public']['epochs'],
                  validation_data=(x_valid, y_valid))
    clf_model.summary()

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def text_dcnn(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    x_valid = input_['x_valid']
    y_valid = input_['y_valid']
    word_indexs = input_['word_indexs']
    parameter = input_['parameter']
    ###########################################

    doc_maxlen = x_train.shape[1]
    clf_model = TextDCNN(input_dim=word_indexs, output_dim=100,
                         input_length=doc_maxlen, output_units=len(set(y_train)),
                         filters_list=parameter['Classifier']['deep_learning']['text_dcnn']['filters_list'],
                         input_shapes=(doc_maxlen,))

    # 模型配置
    clf_model.compile(run_eagerly=True,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
    clf_model.fit(x_train, y_train,
                  batch_size=parameter['Classifier']['deep_learning']['public']['batch_size'],
                  epochs=parameter['Classifier']['deep_learning']['public']['epochs'],
                  validation_data=(x_valid, y_valid))
    clf_model.summary()

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def dpcnn(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    x_valid = input_['x_valid']
    y_valid = input_['y_valid']
    word_indexs = input_['word_indexs']
    parameter = input_['parameter']
    wordvec_model = input_['wordvec_model']
    embedding_weight = input_['embedding_weight']
    ###########################################
    doc_maxlen = x_train.shape[1]

    clf_model = DPCNN(blocks=7, input_dim=word_indexs,
                      output_dim=wordvec_model.dim,
                      input_length=doc_maxlen,
                      embedding_weight=embedding_weight,
                      output_units=len(set(y_train)),
                      input_shapes=(doc_maxlen,))

    # 模型配置
    clf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
    clf_model.fit(x_train, y_train,
                  batch_size=parameter['Classifier']['deep_learning']['public']['batch_size'],
                  epochs=parameter['Classifier']['deep_learning']['public']['epochs'],
                  validation_data=(x_valid, y_valid))
    clf_model.summary()

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def text_inception(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    x_valid = input_['x_valid']
    y_valid = input_['y_valid']
    word_indexs = input_['word_indexs']
    parameter = input_['parameter']
    ###########################################

    doc_maxlen = x_train.shape[1]
    clf_model = TextInception(input_dim=word_indexs,
                              output_dim=100,
                              input_length=doc_maxlen,
                              output_units=len(set(y_train)),
                              input_shapes=(doc_maxlen,))

    # 模型配置
    clf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
    clf_model.fit(x_train, y_train,
                  batch_size=parameter['Classifier']['deep_learning']['public']['batch_size'],
                  epochs=parameter['Classifier']['deep_learning']['public']['epochs'],
                  validation_data=(x_valid, y_valid))
    clf_model.summary()

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def text_vdcnn(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    x_valid = input_['x_valid']
    y_valid = input_['y_valid']
    word_indexs = input_['word_indexs']
    parameter = input_['parameter']
    ###########################################

    doc_maxlen = x_train.shape[1]
    clf_model = TextVDCNN(input_dim=word_indexs,
                          output_dim=100,
                          input_length=doc_maxlen,
                          output_units=len(set(y_train)),
                          input_shapes=(doc_maxlen,),
                          init_filters=parameter['Classifier']['deep_learning']['text_vdcnn']['init_filters'],
                          blocks=parameter['Classifier']['deep_learning']['text_vdcnn']['blocks'])

    # 模型配置
    clf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
    clf_model.fit(x_train, y_train,
                  batch_size=parameter['Classifier']['deep_learning']['public']['batch_size'],
                  epochs=parameter['Classifier']['deep_learning']['public']['epochs'],
                  validation_data=(x_valid, y_valid))
    clf_model.summary()

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def simple_rnn(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    x_valid = input_['x_valid']
    y_valid = input_['y_valid']
    word_indexs = input_['word_indexs']
    parameter = input_['parameter']
    ###########################################
    doc_maxlen = x_train.shape[1]

    clf_model = SimpleRNN(input_dim=word_indexs,
                          output_dim=100,
                          input_length=doc_maxlen,
                          output_units=len(set(y_train)),
                          input_shapes=(doc_maxlen,))

    # 模型配置
    clf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
    clf_model.fit(x_train, y_train,
                  batch_size=parameter['Classifier']['deep_learning']['public']['batch_size'],
                  epochs=parameter['Classifier']['deep_learning']['public']['epochs'],
                  validation_data=(x_valid, y_valid))
    clf_model.summary()

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def lstm(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    x_valid = input_['x_valid']
    y_valid = input_['y_valid']
    word_indexs = input_['word_indexs']
    parameter = input_['parameter']
    ###########################################
    doc_maxlen = x_train.shape[1]

    clf_model = LSTM(input_dim=word_indexs,
                     output_dim=100,
                     input_length=doc_maxlen,
                     output_units=len(set(y_train)),
                     input_shapes=(doc_maxlen,))

    # 模型配置
    clf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
    clf_model.fit(x_train, y_train,
                  batch_size=parameter['Classifier']['deep_learning']['public']['batch_size'],
                  epochs=parameter['Classifier']['deep_learning']['public']['epochs'],
                  validation_data=(x_valid, y_valid))
    clf_model.summary()

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def bi_lstm(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    x_valid = input_['x_valid']
    y_valid = input_['y_valid']
    word_indexs = input_['word_indexs']
    parameter = input_['parameter']
    ###########################################

    doc_maxlen = x_train.shape[1]
    clf_model = BiLSTM(input_dim=word_indexs,
                       output_dim=100,
                       input_length=doc_maxlen,
                       output_units=len(set(y_train)),
                       input_shapes=(doc_maxlen,))

    # 模型配置
    clf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
    clf_model.fit(x_train, y_train,
                  batch_size=parameter['Classifier']['deep_learning']['public']['batch_size'],
                  epochs=parameter['Classifier']['deep_learning']['public']['epochs'],
                  validation_data=(x_valid, y_valid))
    clf_model.summary()

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def gru(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    x_valid = input_['x_valid']
    y_valid = input_['y_valid']
    word_indexs = input_['word_indexs']
    parameter = input_['parameter']
    ###########################################

    doc_maxlen = x_train.shape[1]

    clf_model = GRU(input_dim=word_indexs,
                    output_dim=100,
                    input_length=doc_maxlen,
                    output_units=len(set(y_train)),
                    input_shapes=(doc_maxlen,))

    # 模型配置
    clf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
    clf_model.fit(x_train, y_train,
                  batch_size=parameter['Classifier']['deep_learning']['public']['batch_size'],
                  epochs=parameter['Classifier']['deep_learning']['public']['epochs'],
                  validation_data=(x_valid, y_valid))
    clf_model.summary()

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def bi_gru(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    x_valid = input_['x_valid']
    y_valid = input_['y_valid']
    word_indexs = input_['word_indexs']
    parameter = input_['parameter']
    ###########################################

    doc_maxlen = x_train.shape[1]

    clf_model = BiGRU(input_dim=word_indexs,
                      output_dim=100,
                      input_length=doc_maxlen,
                      output_units=len(set(y_train)),
                      input_shapes=(doc_maxlen,))

    # 模型配置
    clf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
    clf_model.fit(x_train, y_train,
                  batch_size=parameter['Classifier']['deep_learning']['public']['batch_size'],
                  epochs=parameter['Classifier']['deep_learning']['public']['epochs'],
                  validation_data=(x_valid, y_valid))
    clf_model.summary()

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def rcnn(input_):
    ###########################################
    x_train = input_['x_train']
    x_valid = input_['x_valid']

    y_train = input_['y_train']
    y_valid = input_['y_valid']
    word_indexs = input_['word_indexs']
    parameter = input_['parameter']
    ###########################################

    doc_maxlen = x_train.shape[1]

    clf_model = RCNN(input_dim=word_indexs,
                     output_dim=100,
                     input_length=doc_maxlen,
                     output_units=len(set(y_train)))

    # 模型配置
    clf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
    clf_model.fit(x_train, y_train,
                  batch_size=parameter['Classifier']['deep_learning']['public']['batch_size'],
                  epochs=parameter['Classifier']['deep_learning']['public']['epochs'],
                  validation_data=(x_valid, y_valid))
    clf_model.summary()

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def clstm(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    x_valid = input_['x_valid']
    y_valid = input_['y_valid']
    word_indexs = input_['word_indexs']
    parameter = input_['parameter']
    ###########################################

    doc_maxlen = x_train.shape[1]

    clf_model = CLSTM(input_dim=word_indexs,
                      output_dim=100,
                      input_length=doc_maxlen,
                      output_units=len(set(y_train)),
                      input_shapes=(doc_maxlen,))

    # 模型配置
    clf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
    clf_model.fit(x_train, y_train,
                  batch_size=parameter['Classifier']['deep_learning']['public']['batch_size'],
                  epochs=parameter['Classifier']['deep_learning']['public']['epochs'],
                  validation_data=(x_valid, y_valid))
    clf_model.summary()

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def para_clstm(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    x_valid = input_['x_valid']
    y_valid = input_['y_valid']
    word_indexs = input_['word_indexs']
    parameter = input_['parameter']
    ###########################################

    doc_maxlen = x_train.shape[1]

    clf_model = paraCLSTM(input_dim=word_indexs,
                          output_dim=100,
                          input_length=doc_maxlen,
                          output_units=len(set(y_train)),
                          input_shapes=(doc_maxlen,))

    # 模型配置
    clf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
    clf_model.fit(x_train, y_train,
                  batch_size=parameter['Classifier']['deep_learning']['public']['batch_size'],
                  epochs=parameter['Classifier']['deep_learning']['public']['epochs'],
                  validation_data=(x_valid, y_valid))
    clf_model.summary()

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def bilstm_attention(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    x_valid = input_['x_valid']
    y_valid = input_['y_valid']
    word_indexs = input_['word_indexs']
    parameter = input_['parameter']
    ###########################################

    doc_maxlen = x_train.shape[1]
    clf_model = BiLSTMAttention(input_dim=word_indexs,
                                output_dim=100,
                                input_length=doc_maxlen,
                                output_units=len(set(y_train)),
                                input_shapes=(doc_maxlen,))

    # 模型配置
    clf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
    clf_model.fit(x_train, y_train,
                  batch_size=parameter['Classifier']['deep_learning']['public']['batch_size'],
                  epochs=parameter['Classifier']['deep_learning']['public']['epochs'],
                  validation_data=(x_valid, y_valid))
    clf_model.summary()

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def self_attention(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    x_valid = input_['x_valid']
    y_valid = input_['y_valid']
    word_indexs = input_['word_indexs']
    parameter = input_['parameter']
    ###########################################

    doc_maxlen = x_train.shape[1]
    clf_model = SelfAttentionModel(input_dim=word_indexs,
                                   output_dim=100,
                                   input_length=doc_maxlen,
                                   output_units=len(set(y_train)),
                                   input_shapes=(doc_maxlen,))

    # 模型配置
    clf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
    clf_model.fit(x_train, y_train,
                  batch_size=parameter['Classifier']['deep_learning']['public']['batch_size'],
                  epochs=parameter['Classifier']['deep_learning']['public']['epochs'],
                  validation_data=(x_valid, y_valid))
    clf_model.summary()

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def bilstm_selfattention(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    x_valid = input_['x_valid']
    y_valid = input_['y_valid']
    word_indexs = input_['word_indexs']
    parameter = input_['parameter']
    ###########################################
    doc_maxlen = x_train.shape[1]
    clf_model = BiLSTMSelfAttention(input_dim=word_indexs,
                                    output_dim=100,
                                    input_length=doc_maxlen,
                                    output_units=len(set(y_train)),
                                    input_shapes=(doc_maxlen,))
    # 模型配置
    clf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
    clf_model.fit(x_train, y_train,
                  batch_size=parameter['Classifier']['deep_learning']['public']['batch_size'],
                  epochs=parameter['Classifier']['deep_learning']['public']['epochs'],
                  validation_data=(x_valid, y_valid))
    clf_model.summary()

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_