import tensorflow as tf
import numpy as np
import text_model
import vgg16
import ops

class VQA_model:
    def __init__(self, options):
        self.options = options

    def build_model_attention(self):

        options = self.options
        image = tf.placeholder('float32',[ options['batch_size'] , options['img_dim'], options['img_dim'], 3], name = 'image')
        vgg = vgg16.vgg16(image)
        vgg_weights = tf.trainable_variables()
        for weight in vgg_weights:
            print weight.name, weight.get_shape()

        answer = tf.placeholder('float32', [None, self.options['ans_vocab_size']], name = "answer")

        # image features 14 X 14 X 512 ---> 196 X 512
        image_features = vgg.pool5
        print "image fe", image_features
        image_features = tf.nn.l2_normalize(image_features, dim = 3)
        print "image fe2", image_features
        image_features_flat = tf.reshape(image_features, [options['batch_size'], -1, 512 ], name = 'image_features_flat')
        image_features_flat = tf.nn.dropout(image_features_flat, 0.5)

        # sentence embedding 196 --> 196 X 196
        tm = text_model.TextModel(options)
        text_tensors = tm.build_model(train = True)
        source_sentence = text_tensors['source_sentence']
        encoded_sentence = text_tensors['encoded_sentence']
        sentence_embedding = tf.slice(encoded_sentence, [0, options['text_length'] - 1, 0], 
            [-1, -1, -1], name = "bytenet_embedding")
        sentence_embedding = tf.nn.dropout(sentence_embedding, 0.5)
        source_embedding_tiled = tf.tile(sentence_embedding, [1, 14 * 14, 1], name = 'source_embedding_tiled')
        
        # combining 2* (196 * 512) --> 196 * 1024
        combined_features = tf.concat([image_features_flat, source_embedding_tiled], axis = 2)
        # combined_features = tf.concat(2, [image_features_flat, source_embedding_tiled])
        # combined_features = tf.nn.dropout(combined_features, 0.5)
        # CONVOLUTION 1
        combined_features = ops.conv1d(combined_features, 512, name = "attention_conv_1")
        combined_features = tf.nn.relu(combined_features)
        combined_features = tf.nn.dropout(combined_features, 0.5)
        combined_logits = ops.conv1d(combined_features,2, name = "attention_conv_2")
        
        combined_logits_1 = tf.reshape( tf.slice(combined_logits, [0, 0, 0], [-1, -1, 1]), [-1, 14 * 14] )
        combined_logits_2 = tf.reshape( tf.slice(combined_logits, [0, 0, 1], [-1, -1, 1]), [-1, 14 * 14] )
        print "comb1", combined_logits_1
        
        prob1 = tf.reshape(tf.nn.softmax(combined_logits_1), [-1,196, 1])
        prob2 = tf.reshape(tf.nn.softmax(combined_logits_2), [-1,196, 1])

        print prob1
        weighted_image_features = tf.multiply(image_features_flat, prob1) + tf.multiply(image_features_flat, prob2)

        image_features_reduced = tf.reduce_mean(weighted_image_features, 1)
        print "image features reduced", image_features_reduced
        print "image features reduced", sentence_embedding
        # final tensors

        combined_embedding = tf.concat([image_features_reduced, tf.reshape(sentence_embedding,[-1, 512])  ], axis = 1)
        # combined_embedding = tf.concat(1, [image_features_reduced, tf.reshape(sentence_embedding,[-1, 512])  ])
        combined_embedding = tf.nn.dropout(combined_embedding, 0.5)
        
        combined_fc_1 = tf.nn.dropout(ops.fully_connected(combined_embedding, 1024, name = "combined_fc_1"), 0.5)
        logits = ops.fully_connected(combined_fc_1, options['ans_vocab_size'], name = "logits")

        ce = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=answer, name = 'ce')
        answer_probab = tf.nn.softmax(logits, name='answer_probab')
        predictions = tf.argmax(answer_probab,1)
        correct_predictions = tf.equal(tf.argmax(answer_probab,1), tf.argmax(answer,1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        loss = tf.reduce_sum(ce, name = 'loss')

        input_tensors = {
            'image' : image,
            'source_sentence' : source_sentence,
            'answer' : answer
        }

        probability_maps = {
            'map1' : prob1,
            'map2' : prob2
        }

        all_variables = tf.trainable_variables()
        model_variables = [var for var in all_variables if var not in vgg_weights]

        print len(model_variables), len(all_variables), len(vgg_weights)
        var_list = {
            'all_variables' : all_variables,
            'vgg_variables' : vgg_weights,
            'model_variables' : model_variables
        }

        vqa_model = {
            'input_tensors' : input_tensors,
            'var_list' : var_list,
            'probability_maps' : probability_maps,
            'loss' : loss,
            'accuracy' : accuracy,
            'predictions' : predictions,
            'vgg' : vgg
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
    vqa.build_model_attention()

if __name__ == '__main__':
    main()
