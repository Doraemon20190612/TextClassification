from sklearn import decomposition
import numpy as np
from scipy import sparse


def pca(input_):
    ###########################################
    docvec_array = input_['docvec_array']
    docvec_predict_array = input_['docvec_predict_array']
    docvec_array_total = np.concatenate([docvec_array, docvec_predict_array], axis=0)
    parameter = input_['parameter']
    ###########################################

    dimension_reduction = decomposition.PCA(
        n_components=parameter['FeatureDimensionReduce']['linear_decomposition']['pca']['n_components'],
        svd_solver=parameter['FeatureDimensionReduce']['linear_decomposition']['pca']['svd_solver'])
    docvec_array_total = dimension_reduction.fit_transform(docvec_array_total)
    docvec_array = docvec_array_total[:len(docvec_array)]
    docvec_predict_array = docvec_array_total[len(docvec_array):]

    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    return output_


def increment_pca(input_):
    ###########################################
    docvec_array = input_['docvec_array']
    docvec_predict_array = input_['docvec_predict_array']
    docvec_array_total = np.concatenate([docvec_array, docvec_predict_array], axis=0)
    parameter = input_['parameter']
    ###########################################

    dimension_reduction = decomposition.IncrementalPCA(
        n_components=parameter['FeatureDimensionReduce']['linear_decomposition']['increment_pca']['n_components'],
        batch_size=parameter['FeatureDimensionReduce']['linear_decomposition']['increment_pca']['batch_size'])
    docvec_array_total = dimension_reduction.fit_transform(docvec_array_total)
    docvec_array = docvec_array_total[:len(docvec_array)]
    docvec_predict_array = docvec_array_total[len(docvec_array):]

    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    return output_


def kernel_pca(input_):
    ###########################################
    docvec_array = input_['docvec_array']
    docvec_predict_array = input_['docvec_predict_array']
    docvec_array_total = np.concatenate([docvec_array, docvec_predict_array], axis=0)
    parameter = input_['parameter']
    ###########################################

    dimension_reduction = decomposition.KernelPCA(
        n_components=parameter['FeatureDimensionReduce']['linear_decomposition']['kernel_pca']['n_components'],
        kernel=parameter['FeatureDimensionReduce']['linear_decomposition']['kernel_pca']['kernel'])
    docvec_array_total = dimension_reduction.fit_transform(docvec_array_total)
    docvec_array = docvec_array_total[:len(docvec_array)]
    docvec_predict_array = docvec_array_total[len(docvec_array):]

    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    return output_


def sparse_pca(input_):
    ###########################################
    docvec_array = input_['docvec_array']
    docvec_predict_array = input_['docvec_predict_array']
    docvec_array_total = np.concatenate([docvec_array, docvec_predict_array], axis=0)
    parameter = input_['parameter']
    ###########################################

    dimension_reduction = decomposition.SparsePCA(
        n_components=parameter['FeatureDimensionReduce']['linear_decomposition']['sparse_pca']['n_components'],
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


def minibatch_pca(input_):
    ###########################################
    docvec_array = input_['docvec_array']
    docvec_predict_array = input_['docvec_predict_array']
    docvec_array_total = np.concatenate([docvec_array, docvec_predict_array], axis=0)
    parameter = input_['parameter']
    ###########################################

    dimension_reduction = decomposition.MiniBatchSparsePCA(
        n_components=parameter['FeatureDimensionReduce']['linear_decomposition']['minibatch_pca']['n_components'],
        batch_size=parameter['FeatureDimensionReduce']['linear_decomposition']['minibatch_pca']['batch_size'])
    docvec_array_total = dimension_reduction.fit_transform(docvec_array_total)
    docvec_array = docvec_array_total[:len(docvec_array)]
    docvec_predict_array = docvec_array_total[len(docvec_array):]
    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    return output_


def factor_analysis(input_):
    ###########################################
    docvec_array = input_['docvec_array']
    docvec_predict_array = input_['docvec_predict_array']
    docvec_array_total = np.concatenate([docvec_array, docvec_predict_array], axis=0)
    parameter = input_['parameter']
    ###########################################

    dimension_reduction = decomposition.FactorAnalysis(
        n_components=parameter['FeatureDimensionReduce']['linear_decomposition']['factor_analysis']['n_components'])
    docvec_array_total = dimension_reduction.fit_transform(docvec_array_total)
    docvec_array = docvec_array_total[:len(docvec_array)]
    docvec_predict_array = docvec_array_total[len(docvec_array):]
    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    return output_


def svd(input_):
    ###########################################
    docvec_array = input_['docvec_array']
    docvec_predict_array = input_['docvec_predict_array']
    docvec_array_total = np.concatenate([docvec_array, docvec_predict_array], axis=0)
    parameter = input_['parameter']
    ###########################################

    docvec_array_total, s, V = sparse.linalg.svds(sparse.coo_matrix(docvec_array_total).asfptype(),
                                                  k=parameter['FeatureDimensionReduce']['linear_decomposition']['svd'][
                                                      'k'])
    docvec_array = docvec_array_total[:len(docvec_array)]
    docvec_predict_array = docvec_array_total[len(docvec_array):]

    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    return output_


def truncated_svd(input_):
    ###########################################
    docvec_array = input_['docvec_array']
    docvec_predict_array = input_['docvec_predict_array']
    docvec_array_total = np.concatenate([docvec_array, docvec_predict_array], axis=0)
    parameter = input_['parameter']
    ###########################################

    dimension_reduction = decomposition.TruncatedSVD(
        n_components=parameter['FeatureDimensionReduce']['linear_decomposition']['truncated_svd']['n_components'],
        n_iter=parameter['FeatureDimensionReduce']['linear_decomposition']['truncated_svd']['n_iter'],
        random_state=parameter['Public']['random_state'])
    docvec_array_total = dimension_reduction.fit_transform(sparse.csr_matrix(docvec_array_total))
    docvec_array = docvec_array_total[:len(docvec_array)]
    docvec_predict_array = docvec_array_total[len(docvec_array):]

    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    return output_


def lda(input_):
    ###########################################
    docvec_array = input_['docvec_array']
    docvec_predict_array = input_['docvec_predict_array']
    docvec_array_total = np.concatenate([docvec_array, docvec_predict_array], axis=0)
    parameter = input_['parameter']
    ###########################################

    dimension_reduction = decomposition.LatentDirichletAllocation(
        n_components=parameter['FeatureDimensionReduce']['linear_decomposition']['lda']['n_components'],
        learning_method=parameter['FeatureDimensionReduce']['linear_decomposition']['lda']['learning_method'],
        learning_offset=parameter['FeatureDimensionReduce']['linear_decomposition']['lda']['learning_offset'])
    docvec_array_total = dimension_reduction.fit_transform(docvec_array_total)
    docvec_array = docvec_array_total[:len(docvec_array)]
    docvec_predict_array = docvec_array_total[len(docvec_array):]
    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    return output_


def fast_ica(input_):
    ###########################################
    docvec_array = input_['docvec_array']
    docvec_predict_array = input_['docvec_predict_array']
    docvec_array_total = np.concatenate([docvec_array, docvec_predict_array], axis=0)
    parameter = input_['parameter']
    ###########################################

    dimension_reduction = decomposition.FastICA(
        n_components=parameter['FeatureDimensionReduce']['linear_decomposition']['fast_ica']['n_components'])
    docvec_array_total = dimension_reduction.fit_transform(docvec_array_total)
    docvec_array = docvec_array_total[:len(docvec_array)]
    docvec_predict_array = docvec_array_total[len(docvec_array):]
    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    return output_


def nmf(input_):
    ###########################################
    docvec_array = input_['docvec_array']
    docvec_predict_array = input_['docvec_predict_array']
    docvec_array_total = np.concatenate([docvec_array, docvec_predict_array], axis=0)
    parameter = input_['parameter']
    ###########################################

    dimension_reduction = decomposition.NMF(
        n_components=parameter['FeatureDimensionReduce']['linear_decomposition']['nmf']['n_components'],
        init=parameter['FeatureDimensionReduce']['linear_decomposition']['nmf']['init'],
        random_state=parameter['Public']['random_state'])
    docvec_array_total = dimension_reduction.fit_transform(sparse.coo_matrix(docvec_array_total))
    docvec_array = docvec_array_total[:len(docvec_array)]
    docvec_predict_array = docvec_array_total[len(docvec_array):]
    ###########################################
    output_ = input_
    output_['docvec_array'] = docvec_array
    output_['docvec_predict_array'] = docvec_predict_array
    ###########################################
    return output_
