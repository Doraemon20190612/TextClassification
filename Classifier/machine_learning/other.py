from sklearn import gaussian_process


def gaussian_process(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    clf_model = gaussian_process.GaussianProcessClassifier(random_state=parameter['Public']['random_state'])
    clf_model.fit(x_train, y_train)

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_
