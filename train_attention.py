import tensorflow as tf
from Models import VQA_model_attention
import data_loader
import argparse
import numpy as np
from os.path import isfile, join
import utils
import scipy.misc
import gc
import time
from random import shuffle
import shutil
import os
from scipy import misc
import json

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--residual_channels', type=int, default=512,
                       help='residual_channels')  
    parser.add_argument('--data_dir', type=str, default='Data',
                       help='Data directory')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch Size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Batch Size')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Expochs')
    parser.add_argument('--version', type=int, default=2,
                       help='VQA data version')
    parser.add_argument('--debug', type=bool, default=False,
                       help='Debug')
    parser.add_argument('--sample_every', type=int, default=5,
                       help='Debug every x iterations')
    parser.add_argument('--evaluate_every', type=int, default=5,
                       help='Evaluate every x steps')
    parser.add_argument('--resume_model', type=str, default=None,
                       help='Trained Model Path')
    parser.add_argument('--training_log_file', type=str, default='Data/training_log.json',
                       help='Log file for accuracy')
    parser.add_argument('--feature_layer', type=str, default="block4",
                       help='CONV FEATURE LAYER, fc7, pool5 or block4')

    args = parser.parse_args()
    
    print "Reading QA DATA", args.version
    qa_data = data_loader.load_questions_answers(args.version, args.data_dir)
    shuffle(qa_data['training'])
    shuffle(qa_data['validation'])
    qa_data['ans_vocab_rev'] = { qa_data['answer_vocab'][ans] : ans for ans in qa_data['answer_vocab']}
    qa_data['ques_vocab_rev'] = { qa_data['question_vocab'][qw] : qw for qw in qa_data['question_vocab']}
    qa_data['ques_vocab_rev'][0] = ""
    ans_vocab_rev = qa_data['ans_vocab_rev']
    ques_vocab_rev = qa_data['ques_vocab_rev']

    print "Reading conv features"
    conv_features, image_id_list = data_loader.load_conv_features(args.version, 'train', args.feature_layer)
    image_id_map = {image_id_list[i] : i for i in xrange(len(image_id_list))}

    model_options = {
        'question_vocab_size' : len(qa_data['question_vocab']) + 1,
        'residual_channels' : args.residual_channels,
        'ans_vocab_size' : len(qa_data['answer_vocab']),
        'filter_width' : 3,
        'img_dim' : 7,
        'img_channels' : 2048,
        'dilations' : [ 1, 2, 4, 8, 16,
                        1, 2, 4, 8, 16,
                        1, 2, 4, 8, 16
                        ],
        'text_model' : 'lstm',
        'dropout_keep_prob' : 0.7
    }
    
    print "MODEL OPTIONS"
    print model_options
    
    model = VQA_model_attention.VQA_model(model_options)
    model.build_model()
    train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(model.loss)
    model.build_generator(reuse = True)

    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()

    saver = tf.train.Saver()
    if args.resume_model:
        saver.restore(sess, args.resume_model)

    step = 0
    training_log = []
    for epoch in xrange(args.epochs):
        batch_no = 0

        while ((batch_no + 1)*args.batch_size) < len(qa_data['training']):
            start = time.clock()
            question, answer, image_features, image_ids = get_batch(
                batch_no, args.batch_size, qa_data, 
                conv_features, image_id_map, 'train', model_options
                )
            
            _, loss_value = sess.run([train_op, model.loss],
                feed_dict={
                    model.question: question,
                    model.image_features: image_features,
                    model.answer: answer
                }
            )
            end = time.clock()
            print "Time for batch of photos", end - start
            print "Time for one epoch (mins)", len(qa_data['training'])/args.batch_size * (end - start)/60.0
            batch_no += 1
            step += 1
            
            print "LOSS", loss_value, batch_no, step, len(qa_data['training'])/args.batch_size, epoch
            print "****"
            if step % args.sample_every == 0:
                try:
                    shutil.rmtree('Data/samples')
                except:
                    pass
                
                os.makedirs('Data/samples')

                pred_answer, prob1, prob2 = sess.run([model.g_predictions, model.g_prob1, model.g_prob2],
                    feed_dict = {
                        model.g_question : question,
                        model.g_image_features : image_features
                    })
                pred_ans_text = utils.answer_indices_to_text(pred_answer, ans_vocab_rev)
                actual_ans_text = utils.answer_indices_to_text(answer, ans_vocab_rev)
                sample_data = []
                print "Actual vs Prediction"
                for sample_i in range(len(pred_ans_text)):
                    print actual_ans_text[sample_i], pred_ans_text[sample_i]
                    question_text = utils.question_indices_to_text(question[sample_i], ques_vocab_rev)
                    image_array = utils.image_array_from_image_id(image_ids[sample_i], 'train')
                    blend1 = utils.get_blend_map(image_array, prob1[sample_i], overlap = True)
                    blend2 = utils.get_blend_map(image_array, prob2[sample_i], overlap = True)
                    sample_data.append({
                        'question' : question_text,
                        'actual_answer' : actual_ans_text[sample_i],
                        'predicted_answer' : pred_ans_text[sample_i],
                        'image_id' : image_ids[sample_i],
                        'batch_index' : sample_i
                        })
                    misc.imsave('Data/samples/{}_actual_image.jpg'.format(sample_i), image_array)
                    misc.imsave('Data/samples/{}_blend1.jpg'.format(sample_i), blend1)
                    misc.imsave('Data/samples/{}_blend2.jpg'.format(sample_i), blend2)

                f = open('Data/samples/sample.json', 'wb')
                f.write(json.dumps(sample_data))
                f.close()

                shutil.make_archive('Data/samples', 'zip', 'Data/samples')

            if step % args.evaluate_every == 0:
                accuracy = evaluate_model(model, qa_data, args, model_options, sess)
                print "ACCURACY>> ", accuracy, step, epoch
                training_log.append({
                    'step' : step,
                    'epoch' : epoch,
                    'accuracy' : accuracy,
                    })
                f = open(args.training_log_file, 'wb')
                f.write(json.dumps(training_log))
                f.close()
                
                save_path = saver.save(sess, "Data/Models{}/model{}.ckpt".format(args.version, epoch))
        
def evaluate_model(model, qa_data, args, model_options, sess):
    conv_features, image_id_list = data_loader.load_conv_features(args.version, 'val', args.feature_layer)
    image_id_map = {image_id_list[i] : i for i in xrange(len(image_id_list))}
    batch_no = 0
    actual_answers = []
    predicted_answers = []
    while ((batch_no + 1)*args.batch_size) < len(qa_data['validation']):
        question, answer, image_features, image_ids = get_batch(
                batch_no, args.batch_size, qa_data, 
                conv_features, image_id_map, 'val', model_options
                )
        [predicted] = sess.run([model.g_predictions], feed_dict = {
            model.g_question : question,
            model.g_image_features : image_features
        })
        batch_accuracy = np.sum(answer == predicted, dtype = "float32")/args.batch_size
        print "Eavluating", batch_no, len(qa_data['validation'])/args.batch_size, batch_accuracy
        actual_answers += list(answer)
        predicted_answers += list(predicted)

        batch_no += 1

    actual_answers = np.array(actual_answers)
    predicted_answers = np.array(predicted_answers)
    accuracy = np.sum(actual_answers == predicted_answers, dtype = "float32")/len(actual_answers)

    return accuracy



def get_batch(batch_no, batch_size, 
   qa_data, conv_features, image_id_map,
   split, model_options):
    if split == 'train':
        qa = qa_data['training']
    else:
        qa = qa_data['validation']

    img_dim = model_options['img_dim']
    img_channels = model_options['img_channels']

    si = (batch_no * batch_size)%len(qa)
    ei = min(len(qa), si + batch_size)
    n = ei - si
    max_question_length = qa_data['max_question_length']

    sentence_batch = np.ndarray( (n, max_question_length), dtype = 'int32')
    answer_batch = np.zeros(n, dtype = 'int32' )
    conv_batch = np.ndarray((n, img_dim, img_dim, img_channels), dtype = 'float32')
    image_ids = []
    count = 0
    for i in range(si, ei):
        sentence_batch[count] = qa[i]['question']
        answer_batch[count] = qa[i]['answer']
        conv_index = image_id_map[ qa[i]['image_id'] ]
        conv_batch[count] = conv_features[conv_index]
        image_ids.append(qa[i]['image_id']  )
        count += 1
    return sentence_batch, answer_batch, conv_batch, image_ids

if __name__ == '__main__':
    main()