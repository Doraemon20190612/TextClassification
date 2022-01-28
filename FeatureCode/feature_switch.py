from sklearn import ensemble
import numpy as np


def randomtrees_embedding(input_):
    ###########################################
    docvec_array = input_['docvec_array']
    docvec_predict_array = input_['docvec_predict_array']
    docvec_array_total = np.concatenate([docvec_array, docvec_predict_array], axis=0)
    parameter = input_['parameter']
    ###########################################

    dimension_raising = ensemble.RandomTreesEmbedding(
        n_estimators=parameter['FeatureCode']['randomtrees_embedding']['n_estimators'])
    docvec_array_total = dimension_raising.fit_transform(docvec_array_total).toarray()
    docvec_array = docvec_array_total[:len(docvec_array)]
    docvec_predict_array = docvec_array_total[len(docvec_array):]
    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    return output_
