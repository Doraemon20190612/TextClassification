from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def wordvec_sum(input_):
    ###########################################
    text_part_ll = input_['text_part_ll']
    text_predict_part_ll = input_['text_predict_part_ll']
    text_part_ll_total = text_part_ll + text_predict_part_ll
    wordvec_model = input_['wordvec_model']

    ###########################################

    def _wordvec_sum(data_part_ll, wordvec_model):
        embedding_size = len(list(wordvec_model.vector.values())[0])
        docvecs_listoflist = []
        for i in tqdm(data_part_ll):
            docvec_listoflist = []
            if i != []:
                for j in i:
                    try:
                        wordvec = wordvec_model.vector[j]
                        docvec_listoflist.append(wordvec.tolist())
                    except:
                        # print(traceback.print_exc())
                        continue
                if docvec_listoflist != []:
                    docvec_array_i = np.array(docvec_listoflist)
                    docvec_array_sum = docvec_array_i.sum(axis=0)
                    docvecs_listoflist.append(docvec_array_sum.tolist())
                else:
                    docvecs_listoflist.append([float(0) for i in range(embedding_size)])
            else:
                docvecs_listoflist.append([float(0) for i in range(embedding_size)])
        return docvecs_listoflist

    docvec_array_total = np.array(_wordvec_sum(text_part_ll_total, wordvec_model))
    docvec_array = docvec_array_total[:len(text_part_ll)]
    docvec_predict_array = docvec_array_total[len(text_part_ll):]

    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    logging.info('embedding求和计算已完成')
    return output_


def wordvec_avg(input_):
    ###########################################
    text_part_ll = input_['text_part_ll']
    text_predict_part_ll = input_['text_predict_part_ll']
    text_part_ll_total = text_part_ll + text_predict_part_ll
    wordvec_model = input_['wordvec_model']

    ###########################################

    def _wordvec_avg(data_part_ll, wordvec_model):
        embedding_size = len(list(wordvec_model.vector.values())[0])
        docvecs_listoflist = []
        for i in tqdm(data_part_ll):
            docvec_listoflist = []
            if i != []:
                for j in i:
                    try:
                        wordvec = wordvec_model.vector[j]
                        docvec_listoflist.append(wordvec.tolist())
                    except:
                        # print(traceback.print_exc())
                        continue
                if docvec_listoflist != []:
                    docvec_array_i = np.array(docvec_listoflist)
                    docvec_array_sum = docvec_array_i.sum(axis=0) / len(docvec_array_i)
                    docvecs_listoflist.append(docvec_array_sum.tolist())
                else:
                    docvecs_listoflist.append([float(0) for i in range(embedding_size)])
            else:
                docvecs_listoflist.append([float(0) for i in range(embedding_size)])
        return docvecs_listoflist

    docvec_array_total = np.array(_wordvec_avg(text_part_ll_total, wordvec_model))
    docvec_array = docvec_array_total[:len(text_part_ll)]
    docvec_predict_array = docvec_array_total[len(text_part_ll):]

    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    logging.info('embedding平均计算已完成')
    return output_


def wordvec_sif_avg(input_):
    ###########################################
    text_part_ll = input_['text_part_ll']
    text_predict_part_ll = input_['text_predict_part_ll']
    text_part_ll_total = text_part_ll + text_predict_part_ll
    wordvec_model = input_['wordvec_model']

    ###########################################

    def _wordvec_sif_avg(data_part_ll, wordvec_model):
        logging.info('生成SIF权重向量')
        embedding_size = len(list(wordvec_model.vector.values())[0])
        alpha = 0.001
        corpus_size = 0
        v = len(wordvec_model.count)
        pw = np.zeros(v, dtype=np.float32)
        for word in tqdm(wordvec_model.count):
            c = wordvec_model.count[word]
            corpus_size += c
            pw[wordvec_model.index[word]] = c
        pw /= corpus_size
        word_weights = (alpha / (alpha + pw)).astype(np.float32)

        logging.info('进行SIF加权平均')
        docvecs_listoflist = []
        for i in tqdm(data_part_ll):
            docvec_listoflist = []
            word_weights_sum = 0
            if i != []:
                for j in i:
                    try:
                        wordvec = wordvec_model.vector[j] * word_weights[wordvec_model.index[j]]  # SIF加权word2vec
                        word_weights_sum += word_weights[wordvec_model.index[j]]
                        docvec_listoflist.append(wordvec.tolist())
                    except:
                        # print(traceback.print_exc())
                        continue
                if docvec_listoflist != []:
                    docvec_array_i = np.array(docvec_listoflist)
                    docvec_array_sum = docvec_array_i.sum(axis=0) / word_weights_sum
                    docvecs_listoflist.append(docvec_array_sum.tolist())
                else:
                    docvecs_listoflist.append([float(0) for i in range(embedding_size)])
            else:
                docvecs_listoflist.append([float(0) for i in range(embedding_size)])
        return docvecs_listoflist

    docvec_array_total = np.array(_wordvec_sif_avg(text_part_ll_total, wordvec_model))
    docvec_array = docvec_array_total[:len(text_part_ll)]
    docvec_predict_array = docvec_array_total[len(text_part_ll):]
    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    logging.info('embedding sif加权平均计算已完成')
    return output_


def wordvec_usif_avg(input_):
    ###########################################
    text_part_ll = input_['text_part_ll']
    text_predict_part_ll = input_['text_predict_part_ll']
    text_part_ll_total = text_part_ll + text_predict_part_ll
    wordvec_model = input_['wordvec_model']

    ###########################################

    def _wordvec_usif_avg(data_part_ll, wordvec_model):
        logging.info('生成uSIF权重向量')
        embedding_size = len(list(wordvec_model.vector.values())[0])
        corpus_size = 0
        v = len(wordvec_model.count)
        pw = np.zeros(v, dtype=np.float32)
        for word in tqdm(wordvec_model.count):
            c = wordvec_model.count[word]
            corpus_size += c
            pw[wordvec_model.index[word]] = c
        pw /= corpus_size

        length = int(corpus_size / len(data_part_ll))
        threshold = 1 - (1 - (1 / v)) ** length
        alpha = sum(pw > threshold) / v
        z = v / 2
        a = (1 - alpha) / (alpha * z)
        word_weights = (a / (a + pw)).astype(np.float32)

        logging.info('进行uSIF加权平均')
        docvecs_listoflist = []
        for i in tqdm(data_part_ll):
            docvec_listoflist = []
            word_weights_sum = 0
            if i != []:
                for j in i:
                    try:
                        wordvec = wordvec_model.vector[j] * word_weights[wordvec_model.index[j]]  # SIF加权word2vec
                        word_weights_sum += word_weights[wordvec_model.index[j]]
                        docvec_listoflist.append(wordvec.tolist())
                    except:
                        # print(traceback.print_exc())
                        continue
                if docvec_listoflist != []:
                    docvec_array_i = np.array(docvec_listoflist)
                    docvec_array_sum = docvec_array_i.sum(axis=0) / word_weights_sum
                    docvecs_listoflist.append(docvec_array_sum.tolist())
                else:
                    docvecs_listoflist.append([float(0) for i in range(embedding_size)])
            else:
                docvecs_listoflist.append([float(0) for i in range(embedding_size)])
        return np.array(docvecs_listoflist).astype(np.float32)

    docvec_array_total = _wordvec_usif_avg(text_part_ll_total, wordvec_model)
    docvec_array = docvec_array_total[:len(text_part_ll)]
    docvec_predict_array = docvec_array_total[len(text_part_ll):]
    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    logging.info('embedding usif加权平均计算已完成')
    return output_


def wordvec_tfidf_avg(input_):
    ###########################################
    text_part_ll = input_['text_part_ll']
    text_part_sl = input_['text_part_sl']
    text_predict_part_ll = input_['text_predict_part_ll']
    text_predict_part_sl = input_['text_predict_part_sl']
    text_part_ll_total = text_part_ll + text_predict_part_ll
    text_part_sl_total = text_part_sl + text_predict_part_sl
    wordvec_model = input_['wordvec_model']
    parameter = input_['parameter']

    ###########################################

    def _wordvec_tfidf_avg(data_part_ll, data_part_sl, wordvec_model, min_df=1, ngram_range=(1, 2), norm='l1'):
        logging.info('生成tfidf权重')
        embedding_size = len(list(wordvec_model.vector.values())[0])
        countvec = CountVectorizer(min_df=min_df, ngram_range=ngram_range)
        x = countvec.fit_transform(data_part_sl)
        word_list = countvec.get_feature_names()

        tfidf = TfidfTransformer(norm=norm)
        x_tf = tfidf.fit_transform(x)
        word_weights = x_tf.toarray()

        logging.info('进行TFIDF加权平均')
        docvecs_listoflist = []
        for i in tqdm(data_part_ll):
            docvec_listoflist = []
            word_weights_sum = 0
            if i != []:
                for j in i:
                    try:
                        wordvec = wordvec_model.vector[j] * word_weights[data_part_ll.index(i)][
                            word_list.index(j)]  # TFIDF加权word2vec
                        word_weights_sum += word_weights[word_list.index(j)]
                        docvec_listoflist.append(wordvec.tolist())
                    except:
                        # print(traceback.print_exc())
                        continue
                if docvec_listoflist != []:
                    docvec_array_i = np.array(docvec_listoflist)
                    docvec_array_sum = docvec_array_i.sum(axis=0) / word_weights_sum
                    docvecs_listoflist.append(docvec_array_sum.tolist())
                else:
                    docvecs_listoflist.append([float(0) for i in range(embedding_size)])
            else:
                docvecs_listoflist.append([float(0) for i in range(embedding_size)])
        return np.array(docvecs_listoflist)

    docvec_array_total = _wordvec_tfidf_avg(text_part_ll_total, text_part_sl_total, wordvec_model,
                                            min_df=parameter['TextVector']['embedding_process']['wordvec_tfidf_avg'][
                                                'min_df'],
                                            ngram_range=(
                                            parameter['TextVector']['embedding_process']['wordvec_tfidf_avg'][
                                                'ngram_range'][0],
                                            parameter['TextVector']['embedding_process']['wordvec_tfidf_avg'][
                                                'ngram_range'][1]),
                                            norm=parameter['TextVector']['embedding_process']['wordvec_tfidf_avg'][
                                                'norm'])
    docvec_array = docvec_array_total[:len(text_part_ll)]
    docvec_predict_array = docvec_array_total[len(text_part_ll):]
    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    logging.info('embedding tfidf加权平均计算已完成')
    return output_


def wordvec_index(input_):
    ###########################################
    wordvec_model = input_['wordvec_model']
    ###########################################

    embedding_matrix = np.zeros((len(wordvec_model.index) + 1, wordvec_model.dim))
    for w, i in tqdm(wordvec_model.index.items()):
        try:
            embedding_vector = wordvec_model.vector[str(w)]
            embedding_matrix[i + 1] = embedding_vector
        except KeyError:
            continue

    ###########################################
    output_ = input_
    output_['embedding_weight'] = embedding_matrix
    ###########################################
    logging.info('embedding权重矩阵加载已完成')
    return output_
