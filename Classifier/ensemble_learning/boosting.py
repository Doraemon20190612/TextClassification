from sklearn import ensemble
import lightgbm
import catboost
import xgboost
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def ada_boost(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    clf_model = ensemble.AdaBoostClassifier(
        n_estimators=parameter['Classifier']['ensemble_learning']['ada_boost']['n_estimators'],
        random_state=parameter['Public']['random_state'])
    clf_model.fit(x_train, y_train)

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def gbdt(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    clf_model = ensemble.GradientBoostingClassifier(
        n_estimators=parameter['Classifier']['ensemble_learning']['gbdt']['n_estimators'],
        learning_rate=parameter['Classifier']['ensemble_learning']['gbdt']['learning_rate'],
        max_depth=parameter['Classifier']['ensemble_learning']['gbdt']['max_depth'],
        random_state=parameter['Public']['random_state'])
    clf_model.fit(x_train, y_train)

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def hist_gbdt(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    ###########################################

    clf_model = ensemble.HistGradientBoostingClassifier()
    clf_model.fit(x_train, y_train)

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def xgboost(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    clf_model = xgboost.XGBClassifier(
        n_estimators=parameter['Classifier']['ensemble_learning']['xgboost']['n_estimators'],
        max_depth=parameter['Classifier']['ensemble_learning']['xgboost']['max_depth'],
        learning_rate=parameter['Classifier']['ensemble_learning']['xgboost']['learning_rate'])
    clf_model.fit(x_train, y_train)

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_


def light_gbm(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    clf_model = lightgbm.LGBMClassifier(
        n_estimators=parameter['Classifier']['ensemble_learning']['light_gbm']['n_estimators'],
        max_depth=parameter['Classifier']['ensemble_learning']['light_gbm']['max_depth'],
        learning_rate=parameter['Classifier']['ensemble_learning']['light_gbm']['learning_rate'])
    clf_model.fit(x_train, y_train)

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    logging.info('LightGBM训练已完成')
    return output_


def cat_boost(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    clf_model = catboost.CatBoostClassifier(
        eval_metric=parameter['Classifier']['ensemble_learning']['cat_boost']['eval_metric'],
        one_hot_max_size=parameter['Classifier']['ensemble_learning']['cat_boost']['one_hot_max_size'],
        depth=parameter['Classifier']['ensemble_learning']['cat_boost']['depth'],
        iterations=parameter['Classifier']['ensemble_learning']['cat_boost']['iterations'],
        l2_leaf_reg=parameter['Classifier']['ensemble_learning']['cat_boost']['l2_leaf_reg'],
        learning_rate=parameter['Classifier']['ensemble_learning']['cat_boost']['learning_rate'])
    clf_model.fit(x_train, y_train)

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_
