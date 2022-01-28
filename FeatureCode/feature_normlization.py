from sklearn import preprocessing
import numpy as np


def stand_scaler(input_):
    ###########################################
    docvec_array = input_['docvec_array']
    docvec_predict_array = input_['docvec_predict_array']
    docvec_array_total = np.concatenate([docvec_array, docvec_predict_array], axis=0)
    parameter = input_['parameter']
    ###########################################

    standard = preprocessing.StandardScaler(with_mean=parameter['FeatureCode']['stand_scaler']['with_mean'],
                                            with_std=parameter['FeatureCode']['stand_scaler']['with_std'])
    docvec_array_total = standard.fit_transform(docvec_array_total)
    docvec_array = docvec_array_total[:len(docvec_array)]
    docvec_predict_array = docvec_array_total[len(docvec_array):]
    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    return output_