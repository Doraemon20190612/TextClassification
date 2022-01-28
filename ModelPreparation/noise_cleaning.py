from imblearn import under_sampling


def edite_nearest_neighbours(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    sample_model = under_sampling.EditedNearestNeighbours(
        n_neighbors=parameter['ModelPreparation']['noisy_clean']['edite_nearest_neighbours']['n_neighbors'],
        kind_sel=parameter['ModelPreparation']['noisy_clean']['edite_nearest_neighbours']['kind_sel'])
    x_resampled, y_resampled = sample_model.fit_resample(x_train, y_train)

    ###########################################
    output_ = input_
    output_['x_train'] = x_resampled
    output_['y_train'] = y_resampled
    ###########################################
    return output_


def all_knn(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    sample_model = under_sampling.AllKNN(
        n_neighbors=parameter['ModelPreparation']['noisy_clean']['all_knn']['n_neighbors'],
        kind_sel=parameter['ModelPreparation']['noisy_clean']['all_knn']['kind_sel'])
    x_resampled, y_resampled = sample_model.fit_resample(x_train, y_train)

    ###########################################
    output_ = input_
    output_['x_train'] = x_resampled
    output_['y_train'] = y_resampled
    ###########################################
    return output_


def onesided_selection(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    sample_model = under_sampling.OneSidedSelection(
        n_neighbors=parameter['ModelPreparation']['noisy_clean']['onesided_selection']['n_neighbors'],
        n_seeds_S=parameter['ModelPreparation']['noisy_clean']['onesided_selection']['n_seeds_S'],
        random_state=parameter['Public']['random_state'])
    x_resampled, y_resampled = sample_model.fit_resample(x_train, y_train)

    ###########################################
    output_ = input_
    output_['x_train'] = x_resampled
    output_['y_train'] = y_resampled
    ###########################################
    return output_


def neighbourhood_cleaningrule(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    sample_model = under_sampling.NeighbourhoodCleaningRule(
        n_neighbors=parameter['ModelPreparation']['noisy_clean']['neighbourhood_cleaningrule']['n_neighbors'],
        kind_sel=parameter['ModelPreparation']['noisy_clean']['neighbourhood_cleaningrule']['kind_sel'],
        threshold_cleaning=parameter['ModelPreparation']['noisy_clean']['neighbourhood_cleaningrule'][
            'threshold_cleaning'])
    x_resampled, y_resampled = sample_model.fit_resample(x_train, y_train)

    ###########################################
    output_ = input_
    output_['x_train'] = x_resampled
    output_['y_train'] = y_resampled
    ###########################################
    return output_
