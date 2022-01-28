import opencc
import pandas as pd
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def char_converter(input_):
    ###########################################
    data_feature = input_['data_feature']
    data_predict_feature = input_['data_predict_feature']
    parameter = input_['parameter']
    ###########################################

    converter = opencc.OpenCC(config=parameter['TextPreprocess']['char_converter']['config'])
    data = pd.DataFrame(data_feature)
    data.columns = ['text']
    data['text'] = data['text'].apply(lambda x: converter.convert(x))

    data_predict = pd.DataFrame(data_predict_feature)
    data_predict.columns = ['text']
    data_predict['text'] = data_predict['text'].apply(lambda x: converter.convert(x))

    ###########################################
    output_ = input_
    output_['data_feature'] = data['text']
    output_['data_predict_feature'] = data_predict['text']
    ###########################################
    logging.info('文体转换已完成')
    return output_
