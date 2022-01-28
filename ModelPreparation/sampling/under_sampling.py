from imblearn import under_sampling


def cluster_centroids(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    sample_model = under_sampling.ClusterCentroids(random_state=parameter['Public']['random_state'])
    x_resampled, y_resampled = sample_model.fit_resample(x_train, y_train)

    ###########################################
    output_ = input_
    output_['x_train'] = x_resampled
    output_['y_train'] = y_resampled
    ###########################################
    return output_


def random_under_sampling(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    sample_model = under_sampling.RandomUnderSampler(random_state=parameter['Public']['random_state'])
    x_resampled, y_resampled = sample_model.fit_resample(x_train, y_train)

    ###########################################
    output_ = input_
    output_['x_train'] = x_resampled
    output_['y_train'] = y_resampled
    ###########################################
    return output_


def near_miss(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    sample_model = under_sampling.NearMiss(
        version=parameter['ModelPreparation']['data_sample']['under_sample']['near_miss']['version'],
        n_neighbors=parameter['ModelPreparation']['data_sample']['under_sample']['near_miss']['n_neighbors'],
        n_neighbors_ver3=parameter['ModelPreparation']['data_sample']['under_sample']['near_miss']['n_neighbors_ver3'])
    x_resampled, y_resampled = sample_model.fit_resample(x_train, y_train)

    ###########################################
    output_ = input_
    output_['x_train'] = x_resampled
    output_['y_train'] = y_resampled
    ###########################################
    return output_
