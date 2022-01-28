from sklearn import svm


def c_svm(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    clf_model = svm.SVC(C=parameter['Classifier']['machine_learning']['svm']['c_svm']['C'],
                        gamma=parameter['Classifier']['machine_learning']['svm']['c_svm']['gamma'],
                        kernel=parameter['Classifier']['machine_learning']['svm']['c_svm']['kernel'])
    clf_model.fit(x_train, y_train)

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def nu_svm(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    clf_model = svm.NuSVC(nu=parameter['Classifier']['machine_learning']['svm']['nu_svm']['nu'],
                          kernel=parameter['Classifier']['machine_learning']['svm']['nu_svm']['kernel'])
    clf_model.fit(x_train, y_train)

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def linear_svm(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    clf_model = svm.LinearSVC(C=parameter['Classifier']['machine_learning']['svm']['linear_svm']['C'],
                              random_state=parameter['Public']['random_state'])
    clf_model.fit(x_train, y_train)

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_
