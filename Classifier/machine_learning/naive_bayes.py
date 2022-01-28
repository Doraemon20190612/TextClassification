from sklearn import naive_bayes
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def complement_nb(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    clf_model = naive_bayes.ComplementNB(
        alpha=parameter['Classifier']['machine_learning']['naive_bayes']['complement_nb']['alpha'])
    clf_model.fit(x_train, y_train)

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    logging.info('complement贝叶斯训练已完成')
    return output_


def bernoulli_nb(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    clf_model = naive_bayes.BernoulliNB(
        alpha=parameter['Classifier']['machine_learning']['naive_bayes']['bernoulli_nb']['alpha'])
    clf_model.fit(x_train, y_train)

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    logging.info('Bernoulli贝叶斯训练已完成')
    return output_


def multinomial_nb(input_):
    ###########################################
    x_train = input_['x_train']
    y_train = input_['y_train']
    parameter = input_['parameter']
    ###########################################

    clf_model = naive_bayes.MultinomialNB(
        alpha=parameter['Classifier']['machine_learning']['naive_bayes']['multinomial_nb']['alpha'])
    clf_model.fit(x_train, y_train)

    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_
