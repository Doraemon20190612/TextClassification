import transformers
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def bert_loader(input_):
    x_train = input_['x_train']
    x_valid = input_['x_valid']
    x_test = input_['x_test']
    y_train = input_['y_train']
    y_valid = input_['y_valid']
    y_test = input_['y_test']
    x_predict = np.array([[i] for i in input_['data_predict_feature']])
    parameter = input_['parameter']
    ###########################################
    tokenizer = transformers.BertTokenizer.from_pretrained(
        parameter['Classifier']['transfer_learning']['bert']['model_path']
    )

    def _dataset_map(input_ids, token_type_ids, attention_mask, label_ids):
        return {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}, label_ids

    def _tokenizer(x, y):
        input_ids = []
        token_type_ids = []
        attention_mask = []
        label_ids = y.tolist()
        for row in tqdm(x):
            tokenizer_dict = tokenizer.encode_plus(row[0], add_special_tokens=True, max_length=510, pad_to_max_length=True,
                                                   return_attention_mask=True, truncation=True)
            input_ids.append(tokenizer_dict['input_ids'])
            token_type_ids.append(tokenizer_dict['token_type_ids'])
            attention_mask.append(tokenizer_dict['attention_mask'])

        dataset = tf.data.Dataset.from_tensor_slices((input_ids, token_type_ids, attention_mask, label_ids)).map(_dataset_map)
        return dataset

    train_dataset = _tokenizer(x_train, y_train).shuffle(buffer_size=1000).batch(
        batch_size=parameter['Classifier']['transfer_learning']['public']['batch_size']
    )
    valid_dataset = _tokenizer(x_valid, y_valid).batch(
        batch_size=parameter['Classifier']['transfer_learning']['public']['batch_size']
    )
    test_dataset = _tokenizer(x_test, y_test).batch(
        batch_size=parameter['Classifier']['transfer_learning']['public']['batch_size']
    )
    predict_dataset = _tokenizer(x_predict, np.array([0 for i in range(len(x_predict))])).batch(
        batch_size=parameter['Classifier']['transfer_learning']['public']['batch_size']
    )
    ###########################################
    output_ = input_
    output_['train_dataset'] = train_dataset
    output_['valid_dataset'] = valid_dataset
    output_['test_dataset'] = test_dataset
    output_['predict_dataset'] = predict_dataset
    ###########################################
    logging.info('tokenizer_loader已完成')
    return output_
