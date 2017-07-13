import tensorflow as tf
import numpy as np
import ops
import tensorflow.contrib.layers as tf_ops

class TextModel:
    def __init__(self, options):
        '''
        options
        n_source_quant : quantization channels of source text
        residual_channels : number of channels in internal blocks
        batch_size : Batch Size
        text_length : Number of words in sentece
        encoder_filter_width : Encoder Filter Width
        encoder_dilations : Dilation Factor for decoder layers (list)
        
        '''
        self.options = options

        self.w_source_embedding = tf.get_variable('w_source_embedding', 
            [options['n_source_quant'], 2*options['residual_channels']],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

    def build_model(self, train = True):
        options = self.options
        source_sentence = tf.placeholder('int32', [options['batch_size'], options['text_length']], name = 'sentence')
        souce_embedding = tf.nn.embedding_lookup(self.w_source_embedding, source_sentence)

        encoded_sentence = self.encoder(souce_embedding, train = train)

        return {
            'source_sentence' : source_sentence,
            'encoded_sentence' : encoded_sentence
        }


    def encode_layer(self, input_, dilation, layer_no, last_layer = False, train = True):
        options = self.options
        input_ln = tf_ops.layer_norm(input_, trainable = train)
        relu1 = tf.nn.relu(input_ln, name = 'enc_relu1_layer{}'.format(layer_no))
        conv1 = ops.conv1d(relu1, options['residual_channels'], name = 'enc_conv1d_1_layer{}'.format(layer_no))
        conv1 = tf_ops.layer_norm(conv1, trainable = train)
        relu2 = tf.nn.relu(conv1, name = 'enc_relu2_layer{}'.format(layer_no))
        dilated_conv = ops.dilated_conv1d(relu2, options['residual_channels'], 
            dilation, options['encoder_filter_width'],
            causal = False, 
            name = "enc_dilated_conv_layer{}".format(layer_no)
            )
        dilated_conv = tf_ops.layer_norm(trainable = train)
        relu3 = tf.nn.relu(dilated_conv, name = 'enc_relu3_layer{}'.format(layer_no))
        conv2 = ops.conv1d(relu3, 2 * options['residual_channels'], name = 'enc_conv1d_2_layer{}'.format(layer_no))
        return input_ + conv2

    def encoder(self, input_):
        options = self.options
        curr_input = input_
        for layer_no, dilation in enumerate(self.options['encoder_dilations']):
            layer_output = self.encode_layer(curr_input, dilation, layer_no)
            # ENCODE ONLY TILL THE INPUT LENGTH, conditioning should be 0 beyond that
            # layer_output = tf.mul(layer_output, self.souce_embedding, name = 'layer_{}_output'.format(layer_no))
            curr_input = layer_output
        
        processed_output = tf.nn.relu( ops.conv1d(tf.nn.relu(layer_output), 
            options['residual_channels'], 
            name = 'encoder_post_processing') )
        
        return processed_output


def main():
    options = {
        'n_source_quant' : 1000,
        'residual_channels' : 512,
        'batch_size' : 64,
        'text_length' : 25,
        'encoder_filter_width' : 3,
        'encoder_dilations' : [1, 2, 4, 8, 16,
                          1, 2, 4, 8, 16]
    }

    tm = TextModel(options)
    tm.build_model()

if __name__ == '__main__':
    main()
