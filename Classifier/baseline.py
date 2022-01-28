from sklearn import dummy


def dummy(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    clf_model = dummy.DummyClassifier(strategy=parameter['Classifier']['dummy']['strategy'],
                                      constant=parameter['Classifier']['dummy']['constant'])
    clf_model.fit(x_train, y_train)

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_
