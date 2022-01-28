from sklearn import model_selection
import time
import numpy as np
import random
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def split_train_test(input_):
    ###########################################
    parameter = input_['parameter']
    if parameter['ModelPreparation']['split_train_test']['is_original_data'] == 1:
        docvec_array = np.array([[i] for i in input_['data_feature']])
    elif parameter['ModelPreparation']['split_train_test']['is_original_data'] == 0:
        docvec_array = input_['docvec_array']
    else:
        raise ValueError("is_original_data is bool type")
    label_array = input_['label_array']
    aug_mark = input_['aug_mark']
    ###########################################

    def _split_train_test(docvec_array, label_array, aug_mark, parameter):
        pure_index = []
        aug_index = []
        for i, v in enumerate(list(aug_mark)):
            if v == 0:
                pure_index.append(i)
            else:
                aug_index.append(i)

        docvec_array_pure = docvec_array[[pure_index]]
        label_array_pure = label_array[[pure_index]]

        # 划分训练验证集和测试集(去除测试集数据增强的影响)
        x_train_valid_pure, x_test_pure, y_train_valid_pure, y_test_pure = model_selection.train_test_split(
            docvec_array_pure, label_array_pure,
            test_size=parameter['ModelPreparation']['split_train_test']['test_size'],
            random_state=parameter['Public']['random_state'])
        del docvec_array_pure
        time.sleep(1)

        docvec_array_aug = docvec_array[[aug_index]]
        label_array_aug = label_array[[aug_index]]
        if list(docvec_array_aug) == []:
            x_train_valid = x_train_valid_pure
            y_train_valid = y_train_valid_pure
        else:
            x_train_valid = np.concatenate([x_train_valid_pure, docvec_array_aug], axis=0)
            y_train_valid = np.concatenate([y_train_valid_pure, label_array_aug], axis=0)

        del x_train_valid_pure, docvec_array_aug
        time.sleep(1)

        # 打乱顺序
        index_lst = [i for i in range(len(x_train_valid))]
        random.seed(parameter['Public']['random_state'])
        random.shuffle(index_lst)
        x_train_valid = x_train_valid[index_lst]
        y_train_valid = y_train_valid[index_lst]

        # 划分训练集和验证集
        x_train, x_valid, y_train, y_valid = model_selection.train_test_split(x_train_valid, y_train_valid,
                                                                              test_size=parameter['ModelPreparation'][
                                                                                  'split_train_test']['valid_size'],
                                                                              random_state=parameter['Public'][
                                                                                  'random_state'])
        del x_train_valid
        time.sleep(1)

        return x_train, x_valid, y_train, y_valid, x_test_pure, y_test_pure

    x_train, x_valid, y_train, y_valid, x_test_pure, y_test_pure = _split_train_test(docvec_array, label_array,
                                                                                     aug_mark, parameter)

    ###########################################
    output_ = input_
    output_['x_train'] = x_train
    output_['x_valid'] = x_valid
    output_['y_train'] = y_train
    output_['y_valid'] = y_valid
    output_['x_test'] = x_test_pure
    output_['y_test'] = y_test_pure

    del x_train, x_valid, x_test_pure
    time.sleep(1)
    ###########################################
    logging.info('数据拆分已完成')
    return output_
