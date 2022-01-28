from sklearn import manifold
import numpy as np


def isomap(input_):
    ###########################################
    docvec_array = input_['docvec_array']
    docvec_predict_array = input_['docvec_predict_array']
    docvec_array_total = np.concatenate([docvec_array, docvec_predict_array], axis=0)
    parameter = input_['parameter']
    ###########################################

    dimension_reduction = manifold.Isomap(
        n_components=parameter['FeatureDimensionReduce']['manifold_learning']['isomap']['n_components'],
        n_jobs=parameter['Public']['n_jobs'])
    docvec_array_total = dimension_reduction.fit_transform(docvec_array_total)
    docvec_array = docvec_array_total[:len(docvec_array)]
    docvec_predict_array = docvec_array_total[len(docvec_array):]
    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    return output_


def mds(input_):
    ###########################################
    docvec_array = input_['docvec_array']
    docvec_predict_array = input_['docvec_predict_array']
    docvec_array_total = np.concatenate([docvec_array, docvec_predict_array], axis=0)
    parameter = input_['parameter']
    ###########################################

    dimension_reduction = manifold.MDS(
        n_components=parameter['FeatureDimensionReduce']['manifold_learning']['mds']['n_components'],
        n_jobs=parameter['Public']['n_jobs'])
    docvec_array_total = dimension_reduction.fit_transform(docvec_array_total)
    docvec_array = docvec_array_total[:len(docvec_array)]
    docvec_predict_array = docvec_array_total[len(docvec_array):]
    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    return output_


def t_sne(input_):
    ###########################################
    docvec_array = input_['docvec_array']
    docvec_predict_array = input_['docvec_predict_array']
    docvec_array_total = np.concatenate([docvec_array, docvec_predict_array], axis=0)
    parameter = input_['parameter']
    ###########################################

    dimension_reduction = manifold.TSNE(
        n_components=parameter['FeatureDimensionReduce']['manifold_learning']['t_sne']['n_components'],
        n_jobs=parameter['Public']['n_jobs'])
    docvec_array_total = dimension_reduction.fit_transform(docvec_array_total)
    docvec_array = docvec_array_total[:len(docvec_array)]
    docvec_predict_array = docvec_array_total[len(docvec_array):]
    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    return output_


def lle(input_):
    ###########################################
    docvec_array = input_['docvec_array']
    docvec_predict_array = input_['docvec_predict_array']
    docvec_array_total = np.concatenate([docvec_array, docvec_predict_array], axis=0)
    parameter = input_['parameter']
    ###########################################

    docvec_array_total = manifold.locally_linear_embedding(docvec_array_total[:5000],
                                                           n_components=
                                                           parameter['FeatureDimensionReduce']['manifold_learning'][
                                                               'lle']['n_components'],
                                                           n_neighbors=
                                                           parameter['FeatureDimensionReduce']['manifold_learning'][
                                                               'lle']['n_neighbors'],
                                                           n_jobs=parameter['Public']['n_jobs'])[0]
    docvec_array = docvec_array_total[:len(docvec_array)]
    docvec_predict_array = docvec_array_total[len(docvec_array):]
    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    return output_
