from imblearn import combine


def smote_enn(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    sample_model = combine.SMOTEENN(random_state=parameter['Public']['random_state'],
                                    n_jobs=parameter['Public']['n_jobs'])
    x_resampled, y_resampled = sample_model.fit_resample(x_train, y_train)

    ###########################################
    output_ = input_
    output_['x_train'] = x_resampled
    output_['y_train'] = y_resampled
    ###########################################
    return output_


def smote_tomek(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    sample_model = combine.SMOTETomek(random_state=parameter['Public']['random_state'],
                                      n_jobs=parameter['Public']['n_jobs'])
    x_resampled, y_resampled = sample_model.fit_resample(x_train, y_train)

    ###########################################
    output_ = input_
    output_['x_train'] = x_resampled
    output_['y_train'] = y_resampled
    ###########################################
    return output_
