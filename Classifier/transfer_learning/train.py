from .model.bert import BertFineTuning
import tensorflow as tf
import transformers


def bert_finetuning(input_):
    train_dataset = input_['train_dataset']
    valid_dataset = input_['valid_dataset']
    label_array = input_['label_array']
    parameter = input_['parameter']
    ###########################################
    clf_config = transformers.BertConfig.from_pretrained(
        parameter['Classifier']['transfer_learning']['bert']['model_path']
    )
    clf_model = BertFineTuning.from_pretrained(
        parameter['Classifier']['transfer_learning']['bert']['model_path'],
        config=clf_config,
        num_labels=len(set(label_array))
    )
    # 模型配置
    clf_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['sparse_categorical_accuracy'])
    clf_model.fit(train_dataset,
                  epochs=parameter['Classifier']['transfer_learning']['public']['epochs'],
                  validation_data=valid_dataset)
    clf_model.summary()
    ###########################################
    output_ = input_
    output_['clf_model'] = clf_model
    ###########################################
    return output_

