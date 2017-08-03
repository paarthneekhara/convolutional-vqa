import tensorflow as tf
import numpy as np
import ops
from tensorflow.contrib.layers import layer_norm

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
        words_vectors_provided : If True the word Embeddings are provided,
        length_of_word_vector : Lrngth of the Word Vectors Provided/residual_channels
        '''
        self.options = options
        
        if not options['words_vectors_provided']:
            self.w_source_embedding = tf.get_variable('w_source_embedding', 
            [options['n_source_quant'], 2*options['residual_channels']],
            initializer=tf.truncated_normal_initializer(stddev=0.02))


    def build_model(self, train = True):
        options = self.options
        if options['words_vectors_provided']:
            source_sentence = tf.placeholder('float32', [options['batch_size'],options['text_length'], options['length_of_word_vector']], name = 'sentence')
            source_sentence_reshaped = tf.reshape(source_sentence,[-1,300])
            source_embedding = ops.fully_connected(source_sentence_reshaped, 2*options['residual_channels'], name = "tm_source_embedding")
            source_embedding = tf.reshape(source_embedding,[options['batch_size'],options['text_length'],-1])
            print source_embedding
        else:
            source_sentence = tf.placeholder('int32', [options['batch_size'], options['text_length']], name = 'sentence')
            source_embedding = tf.nn.embedding_lookup(self.w_source_embedding, source_sentence)
            print source_embedding
        encoded_sentence = self.encoder(source_embedding, train = train)
        return {
            'source_sentence' : source_sentence,
            'encoded_sentence' : encoded_sentence
        }


    def encode_layer(self, input_, dilation, layer_no, last_layer = False, train = True):
        options = self.options
        input_ = layer_norm(input_, trainable = train)
        relu1 = tf.nn.relu(input_, name = 'enc_relu1_layer{}'.format(layer_no))
        conv1 = ops.conv1d(relu1, options['residual_channels'], name = 'enc_conv1d_1_layer{}'.format(layer_no))
        conv1 = layer_norm(conv1, trainable = train)
        relu2 = tf.nn.relu(conv1, name = 'enc_relu2_layer{}'.format(layer_no))
        dilated_conv = ops.dilated_conv1d(relu2, options['residual_channels'], 
            dilation, options['encoder_filter_width'],
            causal = True, 
            name = "enc_dilated_conv_layer{}".format(layer_no)
            )
        dilated_conv = layer_norm(dilated_conv, trainable = train)
        relu3 = tf.nn.relu(dilated_conv, name = 'enc_relu3_layer{}'.format(layer_no))
        conv2 = ops.conv1d(relu3, 2 * options['residual_channels'], name = 'enc_conv1d_2_layer{}'.format(layer_no))
        return input_ + conv2

    def encoder(self, input_, train = True):
        options = self.options
        curr_input = input_

        for layer_no, dilation in enumerate(self.options['encoder_dilations']):
            if train:
                curr_input = tf.nn.dropout(curr_input, 0.5)
            layer_output = self.encode_layer(curr_input, dilation, layer_no, train)
            curr_input = layer_output
        
        if train:
            layer_output = tf.nn.dropout(layer_output, 0.5)

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
        'words_vectors_provided' : True,
        'length_of_word_vector' : 300,
        'encoder_dilations' : [1, 2, 4, 8, 16,
                          1, 2, 4, 8, 16]

    }


    tm = TextModel(options)
    tm.build_model()

if __name__ == '__main__':
    main()
