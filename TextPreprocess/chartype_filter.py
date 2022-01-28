from tqdm import tqdm
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def number_filter(input_):
    ###########################################
    text_part_ll = input_['text_part_ll']
    text_predict_part_ll = input_['text_predict_part_ll']
    parameter = input_['parameter']
    ###########################################

    def _filter_value(value, method='int'):
        try:
            if method == 'int':
                int(value)
            elif method == 'float':
                float(value)
            return True
        except ValueError:
            return False

    def _data_filter(data_part_ll, method='int'):
        result_ll = []
        for i in tqdm(data_part_ll):
            result_l = [elem for elem in i if not _filter_value(elem, method)]
            result_ll.append(result_l)
        result_sl = [' '.join(j) for j in result_ll]
        return result_ll, result_sl

    text_part_ll, text_part_sl = _data_filter(text_part_ll, method=parameter['TextPreprocess']['number_filter']['method'])
    text_predict_part_ll, text_predict_part_sl = _data_filter(text_predict_part_ll,
                                                              method=parameter['TextPreprocess']['number_filter']['method'])
    ###########################################
    output_ = input_
    output_['text_part_ll'] = text_part_ll
    output_['text_part_sl'] = text_part_sl
    output_['text_predict_part_ll'] = text_predict_part_ll
    output_['text_predict_part_sl'] = text_predict_part_sl
    ###########################################
    logging.info('特定字符过滤已完成')
    return output_