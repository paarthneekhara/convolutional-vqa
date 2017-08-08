import tensorflow as tf
from Models import VQA_model_fc7, VQA_model_lstm
import data_loader
import argparse
import numpy as np
from os.path import isfile, join
import utils
import scipy.misc
import gc
import time
from random import shuffle

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--fc7_feature_length', type=int, default=4096,
                       help='fc7_feature_length')
    parser.add_argument('--residual_channels', type=int, default=512,
                       help='residual_channels')  
    parser.add_argument('--data_dir', type=str, default='Data',
                       help='Data directory')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch Size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Batch Size')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Expochs')
    parser.add_argument('--version', type=int, default=2,
                       help='VQA data version')
    parser.add_argument('--debug', type=bool, default=False,
                       help='Debug')
    parser.add_argument('--debug_every', type=int, default=50,
                       help='Debug every x iterations')
    parser.add_argument('--resume_model', type=str, default=None,
                       help='Trained Model Path')
    parser.add_argument('--text_model', type=str, default='bytenet',
                       help='Text model to choose : butenet/lstm')

    args = parser.parse_args()
    
    print "Reading QA DATA", args.version
    qa_data = data_loader.load_questions_answers(args.version, args.data_dir)
    shuffle(qa_data['training'])
    shuffle(qa_data['validation'])

    print "Reading fc7 features"
    fc7_features, image_id_list = data_loader.load_fc7_features(args.data_dir, 'vgg', args.version, 'train')
    print "FC7 features", fc7_features.shape
    print "image_id_list", image_id_list.shape

    image_id_map = {}
    for i in xrange(len(image_id_list)):
        image_id_map[ image_id_list[i] ] = i

    qa_data['ans_vocab_rev'] = { qa_data['answer_vocab'][ans] : ans for ans in qa_data['answer_vocab']}
    qa_data['ques_vocab_rev'] = { qa_data['question_vocab'][qw] : qw for qw in qa_data['question_vocab']}
    qa_data['ques_vocab_rev'][0] = ""
    ans_vocab_rev = qa_data['ans_vocab_rev']

    model_options = {
        'residual_channels' : args.residual_channels,
        'fc7_feature_length' : args.fc7_feature_length,
        'text_length' : qa_data['max_question_length'],
        'ans_vocab_size' : len(qa_data['answer_vocab']),
        'encoder_filter_width' : 3,
        'batch_size' : args.batch_size,
        'words_vectors_provided' : True,
        'length_of_word_vector' : 300,
        'encoder_dilations' : [1, 2, 4, 8, 16],
        'num_lstm_layers' : 2,
        'img_dim' : 224,
        'text_model' : args.text_model
    }
    
    print "MODEL OPTIONS"
    print model_options

    if args.text_model == 'bytenet':
        model = VQA_model_fc7.VQA_model(model_options)
    else:
        model = VQA_model_lstm.VQA_model(model_options)
    # input_tensors, probability_maps, t_loss, t_accuracy, t_p, variables = model.build_model()
    vqa_model = model.build_model()

    input_tensors = vqa_model['input_tensors']
    t_loss = vqa_model['loss']
    t_accuracy = vqa_model['accuracy']
    t_p = vqa_model['predictions']
    
    train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(t_loss)
    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()

    saver = tf.train.Saver()
    if args.resume_model:
        saver.restore(sess, args.resume_model)

    gc.collect()
    for i in xrange(args.epochs):
        batch_no = 0

        while ((batch_no + 1)*args.batch_size) < len(qa_data['training']):
            start = time.clock()
            sentence, answer,fc7, sentence_words, answer_words = get_training_batch(batch_no, 
                args.batch_size, qa_data, fc7_features, image_id_map, 'train')
            end = time.clock()
            _, loss_value, accuracy, pred = sess.run([
                train_op, t_loss, t_accuracy, t_p], 
                feed_dict={
                    input_tensors['fc7']:fc7,
                    input_tensors['source_sentence']:sentence,
                    input_tensors['answer']:answer
                }
            )
            end = time.clock()
            print "Time for batch of photos", end - start
            print "Time for one epoch", len(qa_data['training'])/args.batch_size * (end - start)
            batch_no += 1
            if args.debug:
                if batch_no % args.debug_every == 0:
                    save_batch(sentence_words, sentence, answer_words, pred, ans_vocab_rev, 'Data/debug')
                    for idx, p in enumerate(pred):
                        print ans_vocab_rev[p], ans_vocab_rev[ np.argmax(answer[idx])]

            print "Loss", loss_value, batch_no, i
            print "Accuracy", accuracy
            print "---------------"
        
        save_path = saver.save(sess, "Data/Models{}_{}_fc7/model{}.ckpt".format(args.version, args.text_model, i))
        
def save_batch(sentence_batch, sentence,
    answer_batch, predictions, ans_vocab_rev, data_dir):
    line_str = ""
    for i in range(len(sentence_batch)):
        sentence_text = " ".join(str(s) for s in sentence_batch[i])
        answer_text = answer_batch[i]
        line_str += "{} question = {} \n{} answer_actual = {} \n{} answer_pred = {} {} \n{} vector = {} \n".format(
            i, sentence_text, i, answer_text, 0, i, ans_vocab_rev[ predictions[i]] , i, np.array_str(sentence[i]) )

    with open(join(data_dir, "ques_ans.txt"), 'wb') as f:
        f.write(line_str)
        f.close()


def get_training_batch(batch_no, batch_size, 
    qa_data, fc7_features, image_id_map, split):
    qa = None
    if split == 'train':
        qa = qa_data['training']
    else:
        qa = qa_data['validation']

    si = (batch_no * batch_size)%len(qa)
    ei = min(len(qa), si + batch_size)
    n = ei - si
    max_question_length = qa_data['max_question_length']
    
    word_vectors = qa_data['word_vectors']
    word_vectors[0] = np.zeros(300)
    ques_vocab_rev = qa_data['ques_vocab_rev']
    ans_vocab_rev = qa_data['ans_vocab_rev']

    sentence_batch = np.ndarray( (n, max_question_length, 300), dtype = 'float32')
    answer_batch = np.zeros( (n, len(qa_data['answer_vocab'])))
    
    fc7 = np.ndarray( (n,4096) )
    
    sentence_words_batch = []
    answer_words_batch = []

    count = 0
    for i in range(si, ei):
        sentence_words_batch.append([])
        for qwi in xrange(max_question_length):
            sentence_batch[count,qwi,:] =  word_vectors[ qa[i]['question'][qwi] ]
            word =  ques_vocab_rev[ qa[i]['question'][qwi] ]
            sentence_words_batch[count].append(word)

        answer_batch[count, qa[i]['answer']] = 1.0
        answer_word = ans_vocab_rev[ qa[i]['answer'] ]
        answer_words_batch.append( answer_word )
        

        fc7_index = image_id_map[ qa[i]['image_id'] ]
        fc7[count,:] = fc7_features[fc7_index][:]

        count += 1

    return sentence_batch, answer_batch, fc7, sentence_words_batch, answer_words_batch

if __name__ == '__main__':
    main()