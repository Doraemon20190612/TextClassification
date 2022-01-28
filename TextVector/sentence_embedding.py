from gensim.models import doc2vec
import numpy as np
import transformers
from tqdm import tqdm, trange
import math
import gc
import time
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def doc2vector(input_):
    ###########################################
    text_part_ll = input_['text_part_ll']
    text_predict_part_ll = input_['text_predict_part_ll']
    text_part_ll_total = text_part_ll + text_predict_part_ll
    parameter = input_['parameter']

    ###########################################

    def doc2vec_model(data_part_ll, vector_size=100, window=20, min_count=1, iter_=500):
        tag_doc = []
        docvec_lst = []
        for i, doc in enumerate(data_part_ll):
            tag_doc.append(doc2vec.TaggedDocument(doc, [i]))
        preprocess_model = doc2vec.Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, iter=iter_)
        preprocess_model.build_vocab(tag_doc)
        for j in range(len(text_part_ll)):
            docvec_lst.append(list(preprocess_model.docvecs[j]))
        return np.array(docvec_lst)

    docvec_array_total = doc2vec_model(text_part_ll_total,
                                       vector_size=parameter['TextVector']['sentence_embedding']['doc2vector'][
                                           'vector_size'],
                                       window=parameter['TextVector']['sentence_embedding']['doc2vector']['window'],
                                       min_count=parameter['TextVector']['sentence_embedding']['doc2vector']['min_count'],
                                       iter_=parameter['TextVector']['sentence_embedding']['doc2vector']['iter_'])
    docvec_array = docvec_array_total[:len(text_part_ll)]
    docvec_predict_array = docvec_array_total[len(text_part_ll):]
    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    logging.info('doc2vec训练已完成')
    return output_


def bert_cls(input_):
    ###########################################
    data_feature = input_['data_feature']
    data_predict_feature = input_['data_predict_feature']
    data_feature_total = data_feature.append(data_predict_feature)
    parameter = input_['parameter']
    ###########################################

    tokenizer = transformers.BertTokenizer.from_pretrained(
        parameter['Classifier']['transfer_learning']['bert']['model_path']
    )
    bert_model = transformers.TFBertModel.from_pretrained(
        parameter['Classifier']['transfer_learning']['bert']['model_path']
    )

    data_feature_total_list = [i for i in tqdm(data_feature_total)]

    del data_predict_feature, data_feature_total
    gc.collect()
    time.sleep(3)

    cls_embeddings = []
    init_num = 0
    for _ in trange(math.ceil(len(data_feature_total_list) / 100)):
        texts = data_feature_total_list[init_num:init_num+100]
        input_token = tokenizer(texts, add_special_tokens=True, max_length=510,
                                pad_to_max_length=True, return_attention_mask=True, truncation=True,
                                return_tensors='tf')
        output_embedding = bert_model(input_token)[0].numpy()

        for e in output_embedding:
            cls_embeddings.append(e[0].tolist())

        del texts, input_token, output_embedding
        gc.collect()
        time.sleep(2)
        init_num += 100

    docvec_array_total = np.array(cls_embeddings)
    docvec_array = docvec_array_total[:len(data_feature)]
    docvec_predict_array = docvec_array_total[len(data_feature):]
    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    logging.info('bert_cls句向量加载已完成')
    return output_
