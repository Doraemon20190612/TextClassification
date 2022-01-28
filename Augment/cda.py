import logging
import nlpcda
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# 随机同义词替换
def aug_simword(input_):
    ###########################################
    data_feature = input_['data_feature']
    label_array = input_['label_array']
    aug_mark = input_['aug_mark']
    parameter = input_['parameter']
    ###########################################

    data = pd.DataFrame(data_feature)
    data.columns = ['text']
    data['label'] = label_array
    data['aug_mark'] = aug_mark

    aug_model = nlpcda.Similarword(create_num=parameter['Augment']['aug_simword']['create_num'],
                                   change_rate=parameter['Augment']['aug_simword']['change_rate'],
                                   seed=parameter['Public']['random_state'])

    for l in parameter['Augment']['aug_simword']['label_list']:
        logging.info('开始对标签%i进行同义词替换文本增强' % l)
        aug_list = []
        for s in tqdm(data[(data['label'] == l) & (data['aug_mark'] == 0)]['text']):
            s_aug = aug_model.replace(s)
            aug_list.extend(s_aug)
        data_aug = pd.DataFrame(aug_list, columns=['text'])
        data_aug['label'] = l
        data_aug['aug_mark'] = 1
        data = pd.concat([data, data_aug], axis=0).sample(frac=1.)

    ###########################################
    output_ = input_
    output_['data_feature'] = data['text']
    output_['label_array'] = np.array(data['label'])
    output_['aug_mark'] = np.array(list(data['aug_mark']))
    ###########################################
    logging.info('同义词替换文本增强已完成')
    return output_


# 随机等价实体替换
def aug_nerword(input_):
    ###########################################
    data_feature = input_['data_feature']
    label_array = input_['label_array']
    aug_mark = input_['aug_mark']
    parameter = input_['parameter']
    ###########################################

    data = pd.DataFrame(data_feature)
    data.columns = ['text']
    data['label'] = label_array
    data['aug_mark'] = aug_mark

    aug_model = nlpcda.Randomword(create_num=parameter['Augment']['aug_nerword']['create_num'],
                                  change_rate=parameter['Augment']['aug_nerword']['change_rate'],
                                  seed=parameter['Public']['random_state'])

    for l in parameter['Augment']['aug_nerword']['label_list']:
        logging.info('开始对标签%i进行命名实体替换文本增强' % l)
        aug_list = []
        for s in tqdm(data[(data['label'] == l) & (data['aug_mark'] == 0)]['text']):
            s_aug = aug_model.replace(s)
            aug_list.extend(s_aug)
        data_aug = pd.DataFrame(aug_list, columns=['text'])
        data_aug['label'] = l
        data_aug['aug_mark'] = 1
        data = pd.concat([data, data_aug], axis=0).sample(frac=1.)

    ###########################################
    output_ = input_
    output_['data_feature'] = data['text']
    output_['label_array'] = np.array(data['label'])
    output_['aug_mark'] = np.array(list(data['aug_mark']))
    ###########################################
    logging.info('命名实体替换文本增强已完成')
    return output_


# 随机近义字替换
def aug_simchar(input_):
    ###########################################
    data_feature = input_['data_feature']
    label_array = input_['label_array']
    aug_mark = input_['aug_mark']
    parameter = input_['parameter']
    ###########################################

    data = pd.DataFrame(data_feature)
    data.columns = ['text']
    data['label'] = label_array
    data['aug_mark'] = aug_mark

    aug_model = nlpcda.Homophone(create_num=parameter['Augment']['aug_simchar']['create_num'],
                                 change_rate=parameter['Augment']['aug_simchar']['change_rate'],
                                 seed=parameter['Public']['random_state'])

    for l in parameter['Augment']['aug_simchar']['label_list']:
        logging.info('开始对标签%i进行近义字替换文本增强' % l)
        aug_list = []
        for s in tqdm(data[(data['label'] == l) & (data['aug_mark'] == 0)]['text']):
            s_aug = aug_model.replace(s)
            aug_list.extend(s_aug)
        data_aug = pd.DataFrame(aug_list, columns=['text'])
        data_aug['label'] = l
        data_aug['aug_mark'] = 1
        data = pd.concat([data, data_aug], axis=0).sample(frac=1.)

    ###########################################
    output_ = input_
    output_['data_feature'] = data['text']
    output_['label_array'] = np.array(data['label'])
    output_['aug_mark'] = np.array(list(data['aug_mark']))
    ###########################################
    logging.info('近义字替换文本增强已完成')
    return output_


# 随机字删除
def aug_delechar(input_):
    ###########################################
    data_feature = input_['data_feature']
    label_array = input_['label_array']
    aug_mark = input_['aug_mark']
    parameter = input_['parameter']
    ###########################################

    data = pd.DataFrame(data_feature)
    data.columns = ['text']
    data['label'] = label_array
    data['aug_mark'] = aug_mark

    aug_model = nlpcda.RandomDeleteChar(create_num=parameter['Augment']['aug_delechar']['create_num'],
                                        change_rate=parameter['Augment']['aug_delechar']['change_rate'],
                                        seed=parameter['Public']['random_state'])

    for l in parameter['Augment']['aug_delechar']['label_list']:
        logging.info('开始对标签%i进行随机字删除文本增强' % l)
        aug_list = []
        for s in tqdm(data[(data['label'] == l) & (data['aug_mark'] == 0)]['text']):
            s_aug = aug_model.replace(s)
            aug_list.extend(s_aug)
        data_aug = pd.DataFrame(aug_list, columns=['text'])
        data_aug['label'] = l
        data_aug['aug_mark'] = 1
        data = pd.concat([data, data_aug], axis=0).sample(frac=1.)

    ###########################################
    output_ = input_
    output_['data_feature'] = data['text']
    output_['label_array'] = np.array(data['label'])
    output_['aug_mark'] = np.array(list(data['aug_mark']))
    ###########################################
    logging.info('随机字删除文本增强已完成')
    return output_


# 随机置换邻近字
def aug_exchangechar(input_):
    ###########################################
    data_feature = input_['data_feature']
    label_array = input_['label_array']
    aug_mark = input_['aug_mark']
    parameter = input_['parameter']
    ###########################################

    data = pd.DataFrame(data_feature)
    data.columns = ['text']
    data['label'] = label_array
    data['aug_mark'] = aug_mark

    aug_model = nlpcda.CharPositionExchange(create_num=parameter['Augment']['aug_exchangechar']['create_num'],
                                            change_rate=parameter['Augment']['aug_exchangechar']['change_rate'],
                                            char_gram=parameter['Augment']['aug_exchangechar']['char_gram'],
                                            seed=parameter['Public']['random_state'])

    for l in parameter['Augment']['aug_exchangechar']['label_list']:
        logging.info('开始对标签%i进行随机置换邻近字文本增强' % l)
        aug_list = []
        for s in tqdm(data[(data['label'] == l) & (data['aug_mark'] == 0)]['text']):
            s_aug = aug_model.replace(s)
            aug_list.extend(s_aug)
        data_aug = pd.DataFrame(aug_list, columns=['text'])
        data_aug['label'] = l
        data_aug['aug_mark'] = 1
        data = pd.concat([data, data_aug], axis=0).sample(frac=1.)

    ###########################################
    output_ = input_
    output_['data_feature'] = data['text']
    output_['label_array'] = np.array(data['label'])
    output_['aug_mark'] = np.array(list(data['aug_mark']))
    ###########################################
    logging.info('随机置换邻近字文本增强已完成')
    return output_


# 等价字替换
def aug_equchar(input_):
    ###########################################
    data_feature = input_['data_feature']
    label_array = input_['label_array']
    aug_mark = input_['aug_mark']
    parameter = input_['parameter']
    ###########################################

    data = pd.DataFrame(data_feature)
    data.columns = ['text']
    data['label'] = label_array
    data['aug_mark'] = aug_mark

    aug_model = nlpcda.EquivalentChar(create_num=parameter['Augment']['aug_equchar']['create_num'],
                                      change_rate=parameter['Augment']['aug_equchar']['change_rate'],
                                      seed=parameter['Public']['random_state'])

    for l in parameter['Augment']['aug_equchar']['label_list']:
        logging.info('开始对标签%i进行等价字替换文本增强' % l)
        aug_list = []
        for s in tqdm(data[(data['label'] == l) & (data['aug_mark'] == 0)]['text']):
            s_aug = aug_model.replace(s)
            aug_list.extend(s_aug)
        data_aug = pd.DataFrame(aug_list, columns=['text'])
        data_aug['label'] = l
        data_aug['aug_mark'] = 1
        data = pd.concat([data, data_aug], axis=0).sample(frac=1.)

    ###########################################
    output_ = input_
    output_['data_feature'] = data['text']
    output_['label_array'] = np.array(data['label'])
    output_['aug_mark'] = np.array(list(data['aug_mark']))
    ###########################################
    logging.info('等价字替换文本增强已完成')
    return output_


# 文本回译
def aug_backtrans(input_):
    ###########################################
    data_feature = input_['data_feature']
    label_array = input_['label_array']
    aug_mark = input_['aug_mark']
    parameter = input_['parameter']
    ###########################################

    data = pd.DataFrame(data_feature)
    data.columns = ['text']
    data['label'] = label_array
    data['aug_mark'] = aug_mark

    trans_list = parameter['Augment']['aug_backtrans']['trans_list']
    for la in trans_list:
        for l in parameter['Augment']['aug_backtrans']['label_list']:
            logging.info('开始对标签%i进行%s文本回译' % (l, la))
            aug_list = []
            for s in tqdm(data[(data['label'] == l) & (data['aug_mark'] == 0)]['text']):
                s_aug_tmp = nlpcda.baidu_translate(content=s, appid=parameter['Augment']['aug_backtrans']['appid'],
                                                   secretKey=parameter['Augment']['aug_backtrans']['secretKey'],
                                                   t_from='zh', t_to=la)
                time.sleep(1)
                s_aug = nlpcda.baidu_translate(content=s_aug_tmp, appid=parameter['Augment']['aug_backtrans']['appid'],
                                               secretKey=parameter['Augment']['aug_backtrans']['secretKey'],
                                               t_from=la, t_to='zh')
                time.sleep(1)
                aug_list.extend(s_aug)
            data_aug = pd.DataFrame(aug_list, columns=['text'])
            data_aug['label'] = l
            data_aug['aug_mark'] = 1
            data = pd.concat([data, data_aug], axis=0).sample(frac=1.)

    ###########################################
    output_ = input_
    output_['data_feature'] = data['text']
    output_['label_array'] = np.array(data['label'])
    output_['aug_mark'] = np.array(list(data['aug_mark']))
    ###########################################
    logging.info('文本回译已完成')
    return output_
