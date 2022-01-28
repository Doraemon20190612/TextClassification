from sklearn import discriminant_analysis


def linear_discriminant(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    clf_model = discriminant_analysis.LinearDiscriminantAnalysis(
        solver=parameter['Classifier']['machine_learning']['discriminant']['linear_discriminant']['solver'])
    clf_model.fit(x_train, y_train)

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def quadratic_discriminant(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    ###########################################

    clf_model = discriminant_analysis.QuadraticDiscriminantAnalysis()
    clf_model.fit(x_train, y_train)

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_
