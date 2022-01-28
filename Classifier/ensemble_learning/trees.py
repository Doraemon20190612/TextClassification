from sklearn import ensemble


def extra_trees(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    clf_model = ensemble.ExtraTreesClassifier(
        n_estimators=parameter['Classifier']['ensemble_learning']['extra_trees']['n_estimators'],
        random_state=parameter['Public']['random_state'])
    clf_model.fit(x_train, y_train)

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def random_forest(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    clf_model = ensemble.RandomForestClassifier(
        n_estimators=parameter['Classifier']['ensemble_learning']['random_forest']['n_estimators'],
        max_depth=parameter['Classifier']['ensemble_learning']['random_forest']['max_depth'],
        random_state=parameter['Public']['random_state'])
    clf_model.fit(x_train, y_train)

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_



