from sklearn import tree


def decision_tree(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    ###########################################

    clf_model = tree.DecisionTreeClassifier()
    clf_model.fit(x_train, y_train)

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def extra_tree(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    ###########################################

    clf_model = tree.ExtraTreeClassifier()
    clf_model.fit(x_train, y_train)

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_
