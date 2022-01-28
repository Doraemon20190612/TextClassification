from imblearn import over_sampling


def random_over_sampling(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    sample_model = over_sampling.RandomOverSampler(
        shrinkage=parameter['ModelPreparation']['data_sample']['over_sample']['random_over_sampling']['shrinkage'],
        random_state=parameter['Public']['random_state'])
    x_resampled, y_resampled = sample_model.fit_resample(x_train, y_train)

    ###########################################
    output_ = input_
    output_['x_train'] = x_resampled
    output_['y_train'] = y_resampled
    ###########################################
    return output_


def smote(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    sample_model = over_sampling.SMOTE(
        k_neighbors=parameter['ModelPreparation']['data_sample']['over_sample']['smote']['k_neighbors'],
        random_state=parameter['Public']['random_state'])
    x_resampled, y_resampled = sample_model.fit_resample(x_train, y_train)

    ###########################################
    output_ = input_
    output_['x_train'] = x_resampled
    output_['y_train'] = y_resampled
    ###########################################
    return output_


def borderline_smote(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    sample_model = over_sampling.BorderlineSMOTE(
        k_neighbors=parameter['ModelPreparation']['data_sample']['over_sample']['borderline_smote']['k_neighbors'],
        m_neighbors=parameter['ModelPreparation']['data_sample']['over_sample']['borderline_smote']['m_neighbors'],
        random_state=parameter['Public']['random_state'])
    x_resampled, y_resampled = sample_model.fit_resample(x_train, y_train)

    ###########################################
    output_ = input_
    output_['x_train'] = x_resampled
    output_['y_train'] = y_resampled
    ###########################################
    return output_


def smotenc(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    sample_model = over_sampling.SMOTENC(
        k_neighbors=parameter['ModelPreparation']['data_sample']['over_sample']['smotenc']['k_neighbors'],
        random_state=parameter['Public']['random_state'])
    x_resampled, y_resampled = sample_model.fit_resample(x_train, y_train)

    ###########################################
    output_ = input_
    output_['x_train'] = x_resampled
    output_['y_train'] = y_resampled
    ###########################################
    return output_


def svm_smote(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    sample_model = over_sampling.SVMSMOTE(
        k_neighbors=parameter['ModelPreparation']['data_sample']['over_sample']['svm_smote']['k_neighbors'],
        random_state=parameter['Public']['random_state'])
    x_resampled, y_resampled = sample_model.fit_resample(x_train, y_train)

    ###########################################
    output_ = input_
    output_['x_train'] = x_resampled
    output_['y_train'] = y_resampled
    ###########################################
    return output_


def kmeans_smote(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    sample_model = over_sampling.KMeansSMOTE(
        k_neighbors=parameter['ModelPreparation']['data_sample']['over_sample']['kmeans_smote']['k_neighbors'],
        m_neighbors=parameter['ModelPreparation']['data_sample']['over_sample']['kmeans_smote']['m_neighbors'],
        random_state=parameter['Public']['random_state'])
    x_resampled, y_resampled = sample_model.fit_resample(x_train, y_train)

    ###########################################
    output_ = input_
    output_['x_train'] = x_resampled
    output_['y_train'] = y_resampled
    ###########################################
    return output_


def adasyn(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    sample_model = over_sampling.ADASYN(
        k_neighbors=parameter['ModelPreparation']['data_sample']['over_sample']['adasyn']['k_neighbors'],
        random_state=parameter['Public']['random_state'])
    x_resampled, y_resampled = sample_model.fit_resample(x_train, y_train)

    ###########################################
    output_ = input_
    output_['x_train'] = x_resampled
    output_['y_train'] = y_resampled
    ###########################################
    return output_
