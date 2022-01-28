import jieba
import jieba.analyse
from tqdm import tqdm
import math
import collections
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def jieba_tfidf(input_):
    ###########################################
    text_part_sl = input_['text_part_sl']
    text_predict_part_sl = input_['text_predict_part_sl']
    parameter = input_['parameter']

    ###########################################

    def _jieba_tfidf(data_part_sl, topK, withWeight=False):
        text_extracts = []
        for i in tqdm(data_part_sl):
            text_extract = jieba.analyse.extract_tags(i, topK=topK, withWeight=withWeight)
            text_extracts.append(text_extract)
        return text_extracts

    text_extract_ll = _jieba_tfidf(text_part_sl,
                                   topK=parameter['ExtractKeyword']['jieba_tfidf']['topK'],
                                   withWeight=parameter['ExtractKeyword']['jieba_tfidf']['withWeight'])
    text_extract_sl = [' '.join(l) for l in text_extract_ll]
    text_predict_extract_ll = _jieba_tfidf(text_predict_part_sl,
                                           topK=parameter['ExtractKeyword']['jieba_tfidf']['topK'],
                                           withWeight=parameter['ExtractKeyword']['jieba_tfidf']['withWeight'])
    text_predict_extract_sl = [' '.join(l) for l in text_predict_extract_ll]

    ###########################################
    output_ = input_
    output_['text_part_ll'] = text_extract_ll
    output_['text_part_sl'] = text_extract_sl
    output_['text_predict_part_ll'] = text_predict_extract_ll
    output_['text_predict_part_sl'] = text_predict_extract_sl
    ###########################################
    logging.info('基于jieba的idf频率文件TF-IDF已完成')
    return output_


# 训练自有文本的TF-IDF
def define_tfidf(input_):
    ###########################################
    text_part_ll = input_['text_part_ll']
    text_predict_part_ll = input_['text_predict_part_ll']
    parameter = input_['parameter']
    ###########################################

    def self_idf(data_part_ll):
        idf_dic = {}
        for i in range(len(data_part_ll)):
            new_content = data_part_ll[i]
            for word in set(new_content):
                if len(word) > 1:
                    idf_dic[word] = idf_dic.get(word, 0.0) + 1.0  # 包含词条𝑤的文档数
        for k, v in idf_dic.items():
            p = '%.10f' % (math.log(len(data_part_ll) / (v + 1.0)))  # IDF值更新
            idf_dic[k] = p
        return idf_dic

    def self_tf(data_part_ll):
        tf_lst = []
        for i in data_part_ll:
            count_dic = dict(collections.Counter(i))  # 统计文档单个词词频
            count_dic_sum = sum(count_dic.values())  # 统计文档总词频
            for k, v in count_dic.items():
                p = '%.10f' % (v / count_dic_sum)  # TF值更新
                count_dic[k] = p
            tf_lst.append(count_dic)
        return tf_lst

    def self_tfidf(data_part_ll, topK=5, withWeight=False):
        tfidf_lst = []
        idf_dic = self_idf(data_part_ll)
        tf_lst = self_tf(data_part_ll)
        for dic in tqdm(tf_lst):
            for k, v in dic.items():
                p = '%.10f' % (float(v) * float(idf_dic[k]))  # TF-IDF值更新
                dic[k] = float(p)
            dic_sort = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:topK]
            if withWeight == True:
                tfidf_lst.append(dic_sort)  # 输出带tfidf值的关键词列表
            else:
                word_list = []
                for s in dic_sort:
                    word_list.append(s[0])
                tfidf_lst.append(word_list)  # 输出关键词列表
        return tfidf_lst

    text_extract_ll = self_tfidf(text_part_ll,
                                 topK=parameter['ExtractKeyword']['define_tfidf']['topK'],
                                 withWeight=parameter['ExtractKeyword']['define_tfidf']['withWeight'])
    text_extract_sl = [' '.join(l) for l in text_extract_ll]
    text_predict_extract_ll = self_tfidf(text_predict_part_ll,
                                         topK=parameter['ExtractKeyword']['define_tfidf']['topK'],
                                         withWeight=parameter['ExtractKeyword']['define_tfidf']['withWeight'])
    text_predict_extract_sl = [' '.join(l) for l in text_predict_extract_ll]

    ###########################################
    output_ = input_
    output_['text_part_ll'] = text_extract_ll
    output_['text_part_sl'] = text_extract_sl
    output_['text_predict_part_ll'] = text_predict_extract_ll
    output_['text_predict_part_sl'] = text_predict_extract_sl
    ###########################################
    logging.info('基于自定义idf频率文件TF-IDF已完成')
    return output_
