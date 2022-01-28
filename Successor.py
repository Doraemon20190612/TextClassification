import numpy as np
from tqdm import tqdm
from sklearn import metrics
import pandas as pd
import pickle


class ModelEvaluation(object):
    def __init__(self, f1_score='micro'):
        self.f1_score = f1_score

    def _predict(self, model, x_test):
        y_test_predict = model.predict(x_test)
        if len(y_test_predict.shape) == 1:
            return y_test_predict
        else:
            return np.argmax(y_test_predict, axis=1)

    def _predict_prob(self, model, x_test):
        try:
            y_test_predict = model.predict_proba(x_test)
        except:
            y_test_predict = model.predict(x_test)
        return y_test_predict

    def run(self, output_list):
        model_lst = []
        accuracy_lst_valid = []
        micro_f1_lst_valid = []
        auc_lst_valid = []
        accuracy_lst_test = []
        micro_f1_lst_test = []
        auc_lst_test = []
        for i in tqdm(output_list):
            model = i['clf_model']
            model_lst.append(model)

            x_valid = i['x_valid']
            y_valid = i['y_valid']
            y_valid_predict = self._predict(model, x_valid)
            y_valid_predict_prob = self._predict_prob(model, x_valid)
            accuracy_valid = metrics.accuracy_score(y_valid, y_valid_predict)
            micro_f1_valid = metrics.f1_score(y_valid, y_valid_predict, average=self.f1_score)
            auc_valid = metrics.roc_auc_score(y_valid, y_valid_predict_prob, multi_class='ovr')
            accuracy_lst_valid.append(accuracy_valid)
            micro_f1_lst_valid.append(micro_f1_valid)
            auc_lst_valid.append(auc_valid)

            x_test = i['x_test']
            y_test = i['y_test']
            y_test_predict = self._predict(model, x_test)
            y_test_predict_prob = self._predict_prob(model, x_test)
            accuracy_test = metrics.accuracy_score(y_test, y_test_predict)
            micro_f1_test = metrics.f1_score(y_test, y_test_predict, average='micro')
            auc_test = metrics.roc_auc_score(y_test, y_test_predict_prob, multi_class='ovr')
            accuracy_lst_test.append(accuracy_test)
            micro_f1_lst_test.append(micro_f1_test)
            auc_lst_test.append(auc_test)

        result_df = pd.DataFrame({'model': model_lst,
                                  'accuracy_valid': accuracy_lst_valid,
                                  'micro_f1_valid': micro_f1_lst_valid,
                                  'auc_valid': auc_lst_valid,
                                  'accuracy_test': accuracy_lst_test,
                                  'micro_f1_test': micro_f1_lst_test,
                                  'auc_test': auc_lst_test})
        return result_df


class ModelPredict(object):
    def __init__(self, return_type='class'):
        self.return_type = return_type

    def run(self, model, predict_data):
        if self.return_type == 'class':
            output = model.predict(predict_data)
            if len(output.shape) == 1:
                return output
            else:
                return np.argmax(output, axis=1)
        elif self.return_type == 'prob':
            output = model.predict(predict_data)
            if len(output.shape) != 1:
                return output
            else:
                return model.predict_proba(predict_data)
        else:
            raise ValueError('return_type in ("class", "prob")')


class ModelSave(object):
    def __init__(self, save_path, name='define'):
        self.save_path = save_path
        self.name = name

    def run(self, model):
        try:
            with open(self.save_path+'%s.pkl' % self.name, 'wb') as file_model:
                pickle.dump(model, file_model)
            print('模型已保存在%s' % (self.save_path + '%s.pkl' % self.name))
        except:
            print("pkl模型保存方案失败,已切换为keras模型保存方案")
            try:
                model.save(self.save_path + '%s/' % self.name)
                print('模型已保存在%s' % (self.save_path + '/%s/' % self.name))
            except:
                print("keras模型保存方案失败,请检查分类器函数名称")


