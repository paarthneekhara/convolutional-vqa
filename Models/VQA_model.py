import tensorflow as tf
import numpy as np
import text_model

class VQA_model:
    def __init__(self, options):


        # TODO
        self.options = options
        self.w_image_embedding = tf.get_variable('w_image_embedding', 
            [options['fc7_feature_length'], options['residual_channels']],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        self.b_image_embedding = tf.Variable(tf.zeros([options['residual_channels']]), name='b_image_embedding')
        
        self.w_ans =  tf.get_variable('w_ans', 
            [options['residual_channels'], options['ans_vocab_size']],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.b_ans = tf.Variable(tf.zeros([options['ans_vocab_size']]), name='b_ans')


    def build_model(self):
        options = self.options
        
        fc7_features = tf.placeholder('float32',[ None, self.options['fc7_feature_length'] ], name = 'fc7')
        answer = tf.placeholder('float32', [None, self.options['ans_vocab_size']], name = "answer")
        
        tm = text_model.TextModel(options)
        text_tensors = tm.build_model(train = True)
        source_sentence = text_tensors['source_sentence']

        encoded_sentence = text_tensors['encoded_sentence']
        
        encoded_shape = encoded_sentence.get_shape()
        sentence_embedding = tf.slice(encoded_sentence, [0, options['text_length'] - 1, 0], 
            [-1, -1, -1])
        
        sentence_embedding = tf.reshape(sentence_embedding, [64, -1])

        image_embedding = tf.matmul(fc7_features, self.w_image_embedding) + self.b_image_embedding
        image_embedding = tf.nn.tanh(image_embedding)
        
        # combined_embedding = sentence_embedding + image_embedding
        combined_embedding = tf.matmul(sentence_embedding, image_embedding)

        logits = tf.matmul(combined_embedding, self.w_ans) + self.b_ans

        ce = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=answer, name = 'ce')
        answer_probab = tf.nn.softmax(logits, name='answer_probab')
        predictions = tf.argmax(answer_probab,1)
        correct_predictions = tf.equal(tf.argmax(answer_probab,1), tf.argmax(answer,1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        loss = tf.reduce_sum(ce, name = 'loss')

        input_tensors = {
            'fc7' : fc7_features,
            'source_sentence' : source_sentence,
            'answer' : answer
        }

        return input_tensors, loss, accuracy, predictions

    def build_generator(self):
        options = self.options
        fc7_features = tf.placeholder('float32',[ None, self.options['fc7_feature_length'] ], name = 'fc7')
        tm = text_model.TextModel(options)
        text_tensors = tm.build_model(train = False)
        source_sentence = text_tensors['source_sentence']

        encoded_sentence = text_tensors['encoded_sentence']
        
        encoded_shape = encoded_sentence.get_shape()
        sentence_embedding = tf.slice(encoded_sentence, [0, options['text_length'] - 1, 0], 
            [-1, -1, -1])
        
        sentence_embedding = tf.reshape(sentence_embedding, [64, -1])

        image_embedding = tf.matmul(fc7_features, self.w_image_embedding) + self.b_image_embedding
        image_embedding = tf.nn.tanh(image_embedding)

        
        # combined_embedding = sentence_embedding + image_embedding
        combined_embedding = tf.matmul(sentence_embedding, image_embedding)
        
        logits = tf.matmul(combined_embedding, self.w_ans) + self.b_ans
        answer_probab = tf.nn.softmax(logits, name='answer_probab')
        predictions = tf.argmax(answer_probab,1)

        input_tensors = {
            'fc7' : fc7_features,
            'source_sentence' : source_sentence,
        }

        return input_tensors, predictions, answer_probab
        



        

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
        'ans_vocab_size' : 1000

    }

    vqa = VQA_model(options)
    vqa.build_model()

if __name__ == '__main__':
    main()
