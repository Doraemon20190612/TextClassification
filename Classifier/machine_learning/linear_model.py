from sklearn import linear_model


def logistic_regression(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    clf_model = linear_model.LogisticRegression(
        penalty=parameter['Classifier']['machine_learning']['linear']['logistic_regression']['penalty'],
        max_iter=parameter['Classifier']['machine_learning']['linear']['logistic_regression']['max_iter'])
    clf_model.fit(x_train, y_train)

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def ridge(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    ###########################################

    clf_model = linear_model.RidgeClassifier()
    clf_model.fit(x_train, y_train)

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def sgd(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    ###########################################

    clf_model = linear_model.SGDClassifier()
    clf_model.fit(x_train, y_train)

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_
