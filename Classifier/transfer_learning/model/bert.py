import transformers
import tensorflow as tf


class BertFineTuning(transformers.TFBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super(BertFineTuning, self).__init__(config, *inputs, **kwargs)
        self.output_units = config.num_labels

        self.bert_ = transformers.TFBertMainLayer(config, name='bert')
        self.dropout_ = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.output_ = tf.keras.layers.Dense(self.output_units,
                                             kernel_initializer=transformers.modeling_tf_utils.get_initializer(config.initializer_range),
                                             activation='sigmoid', name='output_layer')

    def call(self, inputs, **kwargs):
        bert_outputs = self.bert_(inputs, **kwargs)
        pooled_output = bert_outputs[1]
        drop = self.dropout_(pooled_output, training=kwargs.get('training', False))
        logits = self.output_(drop)
        o = (logits,) + bert_outputs[2:]
        return o

