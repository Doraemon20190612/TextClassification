from tqdm import tqdm
import collections
import os
from pymagnitude import *
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Wordembedding(object):
    class Struct(object):
        def __init__(self, count, index, vector, dim):
            self.count = count
            self.index = index
            self.vector = vector
            self.dim = dim

    def data_proccess(self, data_part_ll, min_count=1):
        # 生成词频字典
        word_list = []
        word_count = {}
        for i in tqdm(data_part_ll):
            for j in i:
                word_list.append(j)
        word_count_temp = dict(collections.Counter(word_list))
        for item in word_count_temp.items():
            if item[1] >= min_count:
                word_count[item[0]] = item[1]
            else:
                pass

        # 生成词索引字典
        word_index = {k: v for v, k in enumerate(word_count)}

        return word_count, word_index

    def word2vec_model(self, data_part_ll, sg=1, window=5, alpha=0.025, size=100, min_count=2, iter=500, epochs=10):
        # 词典预处理
        word_count, word_index = self.data_proccess(data_part_ll, min_count=min_count)
        # 模型训练
        model = word2vec.Word2Vec(sg=sg, window=window, alpha=alpha, size=size, min_count=min_count, iter=iter)
        model.build_vocab(data_part_ll)
        model.train(data_part_ll, total_examples=model.corpus_count, epochs=epochs)
        # 生成词向量字典
        word_vector = {}
        for k in tqdm(word_count):
            word_vector[k] = model.wv[k]
        return self.Struct(word_count, word_index, word_vector, size)

    def tencent_model(self, data_part_ll, model_path):
        # 词典预处理
        word_count, word_index = self.data_proccess(data_part_ll)
        # 模型加载
        vectors = Magnitude(model_path)
        # 生成词向量字典
        word_vector = {}
        for k in tqdm(word_count):
            word_vector[k] = vectors.query(k)
        return self.Struct(word_count, word_index, word_vector, 200)

    def co_oc_model(self, data_part_ll, data_part_sl, size=100, min_count=10, ngram_range=(1, 2)):
        word_count, word_index = self.data_proccess(data_part_ll, min_count=min_count)
        # 词频矩阵
        countvec = CountVectorizer(min_df=min_count, ngram_range=ngram_range)
        x = countvec.fit_transform(data_part_sl)
        xc = (x.T * x)
        # 共现矩阵
        matrix = xc.toarray()

        # SVD
        #         U, s, V = np.linalg.svd(matrix,hermitian=True)
        U, s, V = sparse.linalg.svds(sparse.csr_matrix(matrix).asfptype(), k=size)

        # 生成词向量字典
        word_vector = {}
        for k in tqdm(word_count):
            word_vector[k] = U[word_index[k]].astype(np.float32)
        return self.Struct(word_count, word_index, word_vector, size)

    def lsa_model(self, data_part_ll, data_part_sl, size=100, min_count=2, ngram_range=(1, 2)):
        word_count, word_index = self.data_proccess(data_part_ll, min_count=min_count)
        # 词频矩阵
        countvec = CountVectorizer(min_df=min_count, ngram_range=ngram_range)
        x = countvec.fit_transform(data_part_sl)
        xc = x.T
        # 词-样本矩阵
        matrix = xc.toarray()

        U, s, V = sparse.linalg.svds(sparse.csr_matrix(matrix).asfptype(), k=size)

        # 生成词向量字典
        word_vector = {}
        for k in tqdm(word_count):
            word_vector[k] = U[word_index[k]].astype(np.float32)
        return self.Struct(word_count, word_index, word_vector, size)


def word2vector(input_):
    ###########################################
    text_part_ll = input_['text_part_ll']
    text_predict_part_ll = input_['text_predict_part_ll']
    text_part_ll_total = text_part_ll + text_predict_part_ll
    parameter = input_['parameter']
    ###########################################

    word_embedding = Wordembedding()
    wordvec_model = word_embedding.word2vec_model(text_part_ll_total,
                                                  sg=parameter['TextVector']['word_embedding']['word2vector']['sg'],
                                                  window=parameter['TextVector']['word_embedding']['word2vector'][
                                                      'window'],
                                                  alpha=parameter['TextVector']['word_embedding']['word2vector']['alpha'],
                                                  size=parameter['TextVector']['word_embedding']['word2vector']['size'],
                                                  min_count=parameter['TextVector']['word_embedding']['word2vector'][
                                                      'min_count'],
                                                  iter=parameter['TextVector']['word_embedding']['word2vector']['iter'],
                                                  epochs=parameter['TextVector']['word_embedding']['word2vector'][
                                                      'epochs'])

    ###########################################
    output_ = input_
    output_['wordvec_model'] = wordvec_model
    ###########################################
    logging.info('word2vec训练已完成')
    return output_


def tencent_w2v(input_):
    ###########################################
    text_part_ll = input_['text_part_ll']
    text_predict_part_ll = input_['text_predict_part_ll']
    text_part_ll_total = text_part_ll + text_predict_part_ll
    parameter = input_['parameter']
    ###########################################

    word_embedding = Wordembedding()
    wordvec_model = word_embedding.tencent_model(text_part_ll_total,
                                                 model_path=os.path.abspath(os.path.join(os.getcwd(), "..")) + parameter['TextVector']['word_embedding']['tencent_w2v'][
                                                     'model_path'])

    ###########################################
    output_ = input_
    output_['wordvec_model'] = wordvec_model
    ###########################################
    logging.info('tencent词向量模型加载已完成')
    return output_


def co_occurrence(input_):
    ###########################################
    text_part_ll = input_['text_part_ll']
    text_part_sl = input_['text_part_sl']
    text_predict_part_ll = input_['text_predict_part_ll']
    text_predict_part_sl = input_['text_predict_part_sl']
    text_part_ll_total = text_part_ll + text_predict_part_ll
    text_part_sl_total = text_part_sl + text_predict_part_sl
    parameter = input_['parameter']
    ###########################################

    word_embedding = Wordembedding()
    wordvec_model = word_embedding.co_oc_model(text_part_ll_total, text_part_sl_total,
                                               size=parameter['TextVector']['word_embedding']['co_occurrence']['size'],
                                               min_count=parameter['TextVector']['word_embedding']['co_occurrence'][
                                                   'min_count'])

    ###########################################
    output_ = input_
    output_['wordvec_model'] = wordvec_model
    ###########################################
    logging.info('共现矩阵训练已完成')
    return output_


def lsa_vec(input_):
    ###########################################
    text_part_ll = input_['text_part_ll']
    text_part_sl = input_['text_part_sl']
    text_predict_part_ll = input_['text_predict_part_ll']
    text_predict_part_sl = input_['text_predict_part_sl']
    text_part_ll_total = text_part_ll + text_predict_part_ll
    text_part_sl_total = text_part_sl + text_predict_part_sl
    parameter = input_['parameter']
    ###########################################

    word_embedding = Wordembedding()
    wordvec_model = word_embedding.lsa_model(text_part_ll_total, text_part_sl_total,
                                             size=parameter['TextVector']['word_embedding']['lsa_vec']['size'],
                                             min_count=parameter['TextVector']['word_embedding']['lsa_vec'][
                                                 'min_count'])

    ###########################################
    output_ = input_
    output_['wordvec_model'] = wordvec_model
    ###########################################
    logging.info('LSA词向量训练已完成')
    return output_
