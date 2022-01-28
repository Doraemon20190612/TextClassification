from sklearn import neighbors


def knn_kd(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    clf_model = neighbors.KNeighborsClassifier(
        n_neighbors=parameter['Classifier']['machine_learning']['knn']['knn_kd']['n_neighbors'],
        metric=parameter['Classifier']['machine_learning']['knn']['knn_kd']['metric'],
        p=parameter['Classifier']['machine_learning']['knn']['knn_kd']['p'])
    clf_model.fit(x_train, y_train)

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def knn_radius(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    clf_model = neighbors.RadiusNeighborsClassifier(
        radius=parameter['Classifier']['machine_learning']['knn']['knn_radius']['radius'])
    clf_model.fit(x_train, y_train)

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_
