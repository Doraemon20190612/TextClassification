import jieba
from tqdm import tqdm
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def jieba_textrank(input_):
    ###########################################
    text_part_sl = input_['text_part_sl']
    text_predict_part_sl = input_['text_predict_part_sl']
    parameter = input_['parameter']

    ###########################################

    def _jieba_textrank(data_part_sl, topK, withWeight=False):
        text_extracts = []
        for i in tqdm(data_part_sl):
            text_extract = jieba.analyse.textrank(i, topK=topK, withWeight=withWeight)
            text_extracts.append(text_extract)
        return text_extracts

    text_extract_ll = _jieba_textrank(text_part_sl,
                                      topK=parameter['ExtractKeyword']['jieba_textrank']['topK'],
                                      withWeight=parameter['ExtractKeyword']['jieba_textrank']['withWeight'])
    text_extract_sl = [' '.join(l) for l in text_extract_ll]
    text_predict_extract_ll = _jieba_textrank(text_predict_part_sl,
                                              topK=parameter['ExtractKeyword']['jieba_textrank']['topK'],
                                              withWeight=parameter['ExtractKeyword']['jieba_textrank']['withWeight'])
    text_predict_extract_sl = [' '.join(l) for l in text_predict_extract_ll]

    ###########################################
    output_ = input_
    output_['text_part_ll'] = text_extract_ll
    output_['text_part_sl'] = text_extract_sl
    output_['text_predict_part_ll'] = text_predict_extract_ll
    output_['text_predict_part_sl'] = text_predict_extract_sl
    ###########################################
    logging.info('text_rank已完成')
    return output_

