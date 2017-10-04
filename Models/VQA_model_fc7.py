# OLD CODE. NOT BEING USED
import tensorflow as tf
import numpy as np
import text_model_v2
import vgg16
import ops

class VQA_model:
    def __init__(self, options):
        self.options = options

    def build_model(self, train = True):
        dropout_rate = 1.0
        if train:
            dropout = 0.5

        options = self.options
        fc7_features = tf.placeholder('float32',
            [ None, self.options['fc7_feature_length'] ], 
            name = 'fc7')
        source_sentence = tf.placeholder('float32', 
                [ None, options['text_length'], options['length_of_word_vector']], 
                name = 'sentence')
        answer = tf.placeholder('float32', 
            [ None, self.options['ans_vocab_size']], name = "answer")

        

        image_embedding = ops.fully_connected(fc7_features, 2 * options['residual_channels'], 
            name = "image_embedding")
        image_embedding = tf.nn.dropout( tf.nn.tanh(image_embedding), dropout_rate)
        print "image_embedding", image_embedding
        
        # image_features_flat = tf.nn.dropout(image_features_flat, 0.5)
        if options['text_model'] == "bytenet":
            text_tensors = text_model_v2.encoder_bytenet(source_sentence, options)
        else:
            text_tensors = text_model_v2.encoder_lstm(source_sentence, options, train)

        encoded_sentence = text_tensors['last_seq_element']

        encoded_embedding = ops.fully_connected(encoded_sentence, 2 * options['residual_channels'], 
            name = "encoded_embedding")
        encoded_embedding = tf.nn.dropout( tf.nn.tanh(encoded_embedding), dropout_rate )
        print "encoded_embedding", encoded_embedding

        combined_features = encoded_embedding * image_embedding
        combined_features = tf.nn.dropout( combined_features, dropout_rate)
        print "combined", combined_features
        logits = ops.fully_connected(combined_features, options['ans_vocab_size'], name = "logits")
        print "logits", logits
        ce = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=answer, name = 'ce')
        answer_probab = tf.nn.softmax(logits, name='answer_probab')

        predictions = tf.argmax(answer_probab,1)
        correct_predictions = tf.equal(tf.argmax(answer_probab,1), tf.argmax(answer,1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        loss = tf.reduce_mean(ce, name = 'loss')

        input_tensors = {
            'fc7' : fc7_features,
            'source_sentence' : source_sentence,
            'answer' : answer
        }

        vqa_model = {
            'input_tensors' : input_tensors,
            'loss' : loss,
            'accuracy' : accuracy,
            'predictions' : predictions,
        }
        return vqa_model

def main():
    options = {
        'n_source_quant' : 1000,
        'residual_channels' : 512,
        'batch_size' : 64,
        'text_length' : 25,
        'encoder_filter_width' : 3,
        'encoder_dilations' : [1, 2, 4, 8, 16,
                          1, 2, 4, 8, 16],
        'fc7_feature_length' : 4096,
        'words_vectors_provided' : True,
        'length_of_word_vector' : 300,
        'ans_vocab_size' : 1000,
        'img_dim' : 448
    }

    vqa = VQA_model(options)
    vqa.build_model()

if __name__ == '__main__':
    main()
