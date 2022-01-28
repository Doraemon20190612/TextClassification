import jieba
import jieba.analyse
import thulac
import pkuseg
import LAC
import ltp
import snownlp
from tqdm import tqdm
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def jieba_cut(input_):
    ###########################################
    data = input_['data_feature']
    data_predict = input_['data_predict_feature']
    stop_words = input_['stop_words']
    parameter = input_['parameter']
    ###########################################

    def m_cut(data, stop_words):
        return [w for w in jieba.lcut(data,
                                      cut_all=parameter['TextPreprocess']['segment']['jieba_cut']['cut_all'],
                                      use_paddle=parameter['TextPreprocess']['segment']['jieba_cut']['use_paddle'],
                                      HMM=parameter['TextPreprocess']['segment']['jieba_cut']['HMM'])
                if w not in stop_words and len(w) > 1]

    text_part_ll = [m_cut(w, stop_words) for w in tqdm(data)]
    text_part_sl = [' '.join(i) for i in tqdm(text_part_ll)]

    text_predict_part_ll = [m_cut(w, stop_words) for w in tqdm(data_predict)]
    text_predict_part_sl = [' '.join(i) for i in tqdm(text_predict_part_ll)]

    ###########################################
    output_ = input_
    output_['text_part_ll'] = text_part_ll
    output_['text_part_sl'] = text_part_sl
    output_['text_predict_part_ll'] = text_predict_part_ll
    output_['text_predict_part_sl'] = text_predict_part_sl
    ###########################################
    logging.info('jieba分词已完成')
    return output_


def thulac_cut(input_):
    ###########################################
    data = input_['data_feature']
    data_predict = input_['data_predict_feature']
    stop_words = input_['stop_words']
    parameter = input_['parameter']
    ###########################################

    thu = thulac.thulac(seg_only=parameter['TextPreprocess']['segment']['thulac_cut']['seg_only'])

    def m_cut(data, stop_words):
        return [w for w in thu.cut(data, text=parameter['TextPreprocess']['segment']['thulac_cut']['text']).split(' ')
                if w not in stop_words and len(w) > 1]

    text_part_ll = [m_cut(w, stop_words) for w in tqdm(data)]
    text_part_sl = [' '.join(i) for i in tqdm(text_part_ll)]

    text_predict_part_ll = [m_cut(w, stop_words) for w in tqdm(data_predict)]
    text_predict_part_sl = [' '.join(i) for i in tqdm(text_predict_part_ll)]

    ###########################################
    output_ = input_
    output_['text_part_ll'] = text_part_ll
    output_['text_part_sl'] = text_part_sl
    output_['text_predict_part_ll'] = text_predict_part_ll
    output_['text_predict_part_sl'] = text_predict_part_sl
    ###########################################
    return output_


def pkuseg_cut(input_):
    ###########################################
    data = input_['data_feature']
    data_predict = input_['data_predict_feature']
    stop_words = input_['stop_words']
    parameter = input_['parameter']
    ###########################################

    pku = pkuseg.pkuseg(model_name=parameter['TextPreprocess']['segment']['pkuseg_cut']['model_name'],
                        user_dict=parameter['TextPreprocess']['segment']['pkuseg_cut']['user_dict'],
                        postag=parameter['TextPreprocess']['segment']['pkuseg_cut']['postag'])

    def m_cut(data, stop_words):
        return [w for w in pku.cut(data) if w not in stop_words and len(w) > 1]

    text_part_ll = [m_cut(w, stop_words) for w in tqdm(data)]
    text_part_sl = [' '.join(i) for i in tqdm(text_part_ll)]

    text_predict_part_ll = [m_cut(w, stop_words) for w in tqdm(data_predict)]
    text_predict_part_sl = [' '.join(i) for i in tqdm(text_predict_part_ll)]

    ###########################################
    output_ = input_
    output_['text_part_ll'] = text_part_ll
    output_['text_part_sl'] = text_part_sl
    output_['text_predict_part_ll'] = text_predict_part_ll
    output_['text_predict_part_sl'] = text_predict_part_sl
    ###########################################
    return output_


def lac_cut(input_):
    ###########################################
    data = input_['data_feature']
    data_predict = input_['data_predict_feature']
    stop_words = input_['stop_words']
    parameter = input_['parameter']
    ###########################################

    lac = LAC.LAC(mode=parameter['TextPreprocess']['segment']['lac_cut']['mdoe'])

    def m_cut(data, stop_words):
        return [w for w in lac.run(data) if w not in stop_words and len(w) > 1]

    text_part_ll = [m_cut(w, stop_words) for w in tqdm(data)]
    text_part_sl = [' '.join(i) for i in tqdm(text_part_ll)]

    text_predict_part_ll = [m_cut(w, stop_words) for w in tqdm(data_predict)]
    text_predict_part_sl = [' '.join(i) for i in tqdm(text_predict_part_ll)]

    ###########################################
    output_ = input_
    output_['text_part_ll'] = text_part_ll
    output_['text_part_sl'] = text_part_sl
    output_['text_predict_part_ll'] = text_predict_part_ll
    output_['text_predict_part_sl'] = text_predict_part_sl
    ###########################################
    return output_


def ltp_cut(input_):
    ###########################################
    data = input_['data_feature']
    data_predict = input_['data_predict_feature']
    stop_words = input_['stop_words']
    parameter = input_['parameter']
    ###########################################

    hit = ltp.LTP(model=parameter['TextPreprocess']['segment']['ltp_cut']['model'])

    def m_cut(data, stop_words):
        return [w for w in hit.seg([data])[0][0] if w not in stop_words and len(w) > 1]

    text_part_ll = [m_cut(w, stop_words) for w in tqdm(data)]
    text_part_sl = [' '.join(i) for i in tqdm(text_part_ll)]

    text_predict_part_ll = [m_cut(w, stop_words) for w in tqdm(data_predict)]
    text_predict_part_sl = [' '.join(i) for i in tqdm(text_predict_part_ll)]

    ###########################################
    output_ = input_
    output_['text_part_ll'] = text_part_ll
    output_['text_part_sl'] = text_part_sl
    output_['text_predict_part_ll'] = text_predict_part_ll
    output_['text_predict_part_sl'] = text_predict_part_sl
    ###########################################
    return output_


def snow_cut(input_):
    ###########################################
    data = input_['data_feature']
    data_predict = input_['data_predict_feature']
    stop_words = input_['stop_words']
    parameter = input_['parameter']

    ###########################################

    def m_cut(data, stop_words):
        return [w for w in snownlp.SnowNLP(data).words if w not in stop_words and len(w) > 1]

    text_part_ll = [m_cut(w, stop_words) for w in tqdm(data)]
    text_part_sl = [' '.join(i) for i in tqdm(text_part_ll)]

    text_predict_part_ll = [m_cut(w, stop_words) for w in tqdm(data_predict)]
    text_predict_part_sl = [' '.join(i) for i in tqdm(text_predict_part_ll)]

    ###########################################
    output_ = input_
    output_['text_part_ll'] = text_part_ll
    output_['text_part_sl'] = text_part_sl
    output_['text_predict_part_ll'] = text_predict_part_ll
    output_['text_predict_part_sl'] = text_predict_part_sl
    ###########################################
    return output_

