import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def variance_filter(input_):
    ###########################################
    docvec_array = input_['docvec_array']
    docvec_predict_array = input_['docvec_predict_array']
    docvec_array_total = np.concatenate([docvec_array, docvec_predict_array], axis=0)
    parameter = input_['parameter']

    ###########################################

    def _variance_filter(docvec_array, size=100):
        docvec_var = np.var(docvec_array, axis=0)
        docvec_sort = np.argsort(-docvec_var)
        docvec_result = docvec_array[:, list(docvec_sort)[0:size]]
        return docvec_result

    docvec_array_total = _variance_filter(docvec_array_total,
                                          size=parameter['FeatureDimensionReduce']['filters']['variance_filter'][
                                              'size'])
    docvec_array = docvec_array_total[:len(docvec_array)]
    docvec_predict_array = docvec_array_total[len(docvec_array):]

    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    logging.info('方差滤波器特征筛选已完成')
    return output_
