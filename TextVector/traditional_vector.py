from tqdm import tqdm
import numpy as np
import collections
import itertools
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def one_hot(input_):
    ###########################################
    text_part_sl = input_['text_part_sl']
    text_predict_part_sl = input_['text_predict_part_sl']
    text_part_sl_total = text_part_sl + text_predict_part_sl
    parameter = input_['parameter']

    ###########################################

    def _one_hot(data_part_sl, max_length=10):
        token_index = {}  # 构建数据中所有标记的索引
        for sample in data_part_sl:
            for word in sample.split():  # 用split方法对样本进行分词，实际应用中，可能还需要考虑到标点符号
                if word not in token_index:
                    token_index[word] = len(token_index) + 1  # 为每个唯一单词指定唯一索引，注意我们没有为索引编号0指定单词

        results = np.zeros((len(data_part_sl), max_length, max(token_index.values()) + 1))  # 将结果保存到results中
        for i, sample in tqdm(enumerate(data_part_sl)):
            for j, word in list(enumerate(sample.split()))[:max_length]:
                index = token_index.get(word)
                results[i, j, index] = 1.
        return results

    docvec_array_total = _one_hot(text_part_sl_total,
                                  max_length=parameter['TextVector']['traditional']['one_hot']['max_length'])
    docvec_array = docvec_array_total[:len(text_part_sl)]
    docvec_predict_array = docvec_array_total[len(text_part_sl):]
    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    logging.info('onehot已完成')
    return output_


def one_hot_keras(input_):
    ###########################################
    text_part_ll = input_['text_part_ll']
    text_part_sl = input_['text_part_sl']
    text_predict_part_ll = input_['text_predict_part_ll']
    text_predict_part_sl = input_['text_predict_part_sl']
    text_part_ll_total = text_part_ll + text_predict_part_ll
    text_part_sl_total = text_part_sl + text_predict_part_sl
    parameter = input_['parameter']
    ###########################################
    word_indexs = len(collections.Counter(list(itertools.chain(*text_part_ll_total)))) + 1
    doc_maxlen = max([len(i) for i in text_part_ll_total]) + 2

    def _one_hot_keras(text_part_sl_total, word_indexs, doc_maxlen, padding='post'):
        onehot_doc = [tf.keras.preprocessing.text.one_hot(d, word_indexs) for d in text_part_sl_total]
        docvec_array = tf.keras.preprocessing.sequence.pad_sequences(onehot_doc, maxlen=doc_maxlen, padding=padding)
        return docvec_array

    docvec_array_total = _one_hot_keras(text_part_sl_total, word_indexs, doc_maxlen,
                                        padding=parameter['TextVector']['traditional']['one_hot_keras']['padding'])
    docvec_array = docvec_array_total[:len(text_part_sl)]
    docvec_predict_array = docvec_array_total[len(text_part_sl):]

    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    output_['word_indexs'] = word_indexs
    ###########################################
    logging.info('onehot已完成')
    return output_


def count_vector(input_):
    ###########################################
    text_part_sl = input_['text_part_sl']
    text_predict_part_sl = input_['text_predict_part_sl']
    text_part_sl_total = text_part_sl + text_predict_part_sl
    parameter = input_['parameter']
    ###########################################

    preprocess_model = CountVectorizer(min_df=parameter['TextVector']['traditional']['count_vector']['min_df'],
                                       ngram_range=(
                                       parameter['TextVector']['traditional']['count_vector']['ngram_range'][0],
                                       parameter['TextVector']['traditional']['count_vector']['ngram_range'][1]))
    docvec_array_total = preprocess_model.fit_transform(text_part_sl_total).A
    docvec_array = docvec_array_total[:len(text_part_sl)]
    docvec_predict_array = docvec_array_total[len(text_part_sl):]
    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    logging.info('count_vector已完成')
    return output_


def hash_vector(input_):
    ###########################################
    text_part_sl = input_['text_part_sl']
    text_predict_part_sl = input_['text_predict_part_sl']
    text_part_sl_total = text_part_sl + text_predict_part_sl
    parameter = input_['parameter']
    ###########################################

    preprocess_model = HashingVectorizer(n_features=parameter['TextVector']['traditional']['hash_vector']['n_features'],
                                         ngram_range=(
                                         parameter['TextVector']['traditional']['hash_vector']['ngram_range'][0],
                                         parameter['TextVector']['traditional']['hash_vector']['ngram_range'][1]),
                                         norm=parameter['TextVector']['traditional']['hash_vector']['norm'])
    docvec_array_total = preprocess_model.fit_transform(text_part_sl_total).A
    docvec_array = docvec_array_total[:len(text_part_sl)]
    docvec_predict_array = docvec_array_total[len(text_part_sl):]
    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    logging.info('hash_vector已完成')
    return output_


def tfidf_vector(input_):
    ###########################################
    text_part_sl = input_['text_part_sl']
    text_predict_part_sl = input_['text_predict_part_sl']
    text_part_sl_total = text_part_sl + text_predict_part_sl
    parameter = input_['parameter']
    ###########################################

    preprocess_model = TfidfVectorizer(min_df=parameter['TextVector']['traditional']['tfidf_vector']['min_df'],
                                       max_df=parameter['TextVector']['traditional']['tfidf_vector']['max_df'],
                                       ngram_range=(
                                       parameter['TextVector']['traditional']['tfidf_vector']['ngram_range'][0],
                                       parameter['TextVector']['traditional']['tfidf_vector']['ngram_range'][1]),
                                       use_idf=parameter['TextVector']['traditional']['tfidf_vector']['use_idf'],
                                       smooth_idf=parameter['TextVector']['traditional']['tfidf_vector']['smooth_idf'],
                                       norm=parameter['TextVector']['traditional']['tfidf_vector']['norm'],
                                       max_features=parameter['TextVector']['traditional']['tfidf_vector'][
                                           'max_features'],
                                       strip_accents=parameter['TextVector']['traditional']['tfidf_vector'][
                                           'strip_accents'],
                                       token_pattern=r"(?u)\b\w+\b")
    docvec_array_total = preprocess_model.fit_transform(text_part_sl_total).A
    docvec_array = docvec_array_total[:len(text_part_sl)]
    docvec_predict_array = docvec_array_total[len(text_part_sl):]
    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    logging.info('tfidf vectorizer已完成')
    return output_
