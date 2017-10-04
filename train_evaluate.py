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
import pickle

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
    parser.add_argument('--epochs', type=int, default=25,
                       help='Expochs')
    parser.add_argument('--max_steps', type=int, default=50000,
                       help='max steps, set 1 for evaluating the model')
    parser.add_argument('--version', type=int, default=1,
                       help='VQA data version')
    parser.add_argument('--sample_every', type=int, default=200,
                       help='Debug every x iterations')
    parser.add_argument('--evaluate_every', type=int, default=6000,
                       help='Evaluate every x steps')
    parser.add_argument('--resume_model', type=str, default=None,
                       help='Trained Model Path')
    parser.add_argument('--training_log_file', type=str, default='Data/training_log.json',
                       help='Log file for accuracy')
    parser.add_argument('--feature_layer', type=str, default="block4",
                       help='CONV FEATURE LAYER, fc7, pool5 or block4')
    parser.add_argument('--cnn_model', type=str, default="resnet",
                       help='CNN model')
    parser.add_argument('--text_model', type=str, default="bytenet",
                       help='bytenet/lstm')

    # evaluation_steps = [6000, 12000, 18000, 25000, 30000, 35000, 50000]
    # evaluation_steps = [400, 800, 1200, 1600, 2000, 2400, 2800]
    args = parser.parse_args()
    
    print "Reading QA DATA", args.version
    qa_data = data_loader.load_questions_answers(args.version, args.data_dir)
    shuffle(qa_data['training'])
    shuffle(qa_data['validation'])
    
    ans_vocab_rev = qa_data['index_to_ans']
    ques_vocab_rev = qa_data['index_to_qw']

    print "Reading conv features"
    conv_features, image_id_list = data_loader.load_conv_features('train', args.cnn_model, args.feature_layer)
    # image_id_map = {image_id_list[i] : i for i in xrange(len(image_id_list))}
    image_id_map = {image_id_list[i] : i for i in xrange(len(image_id_list))}
    
    conv_features_val, image_id_list_val = data_loader.load_conv_features('val', args.cnn_model, args.feature_layer)
    image_id_map_val = {image_id_list_val[i] : i for i in xrange(len(image_id_list_val))}

    conv_features = data_loader.load_conv_features('train', args.cnn_model, args.feature_layer, load_image_list = False)

    model_options = {
        'question_vocab_size' : len(qa_data['index_to_qw']),
        'residual_channels' : args.residual_channels,
        'ans_vocab_size' : len(qa_data['index_to_ans']),
        'filter_width' : 3,
        'img_dim' : 14,
        'img_channels' : 2048,
        'dilations' : [ 1, 2, 4, 8,
                        1, 2, 4, 8, 
                       ],
        'text_model' : args.text_model,
        'dropout_keep_prob' : 0.6,
        'max_question_length' : qa_data['max_question_length'],
        'num_answers' : 10
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
        while (batch_no*args.batch_size) < len(qa_data['training']):
            start = time.clock()
            question, answer, image_features, image_ids, _ = get_batch(
                batch_no, args.batch_size, qa_data['training'], 
                conv_features, image_id_map, 'train', model_options
                )
            
            _, loss_value = sess.run([train_op, model.loss],
                feed_dict={
                    model.question: question,
                    model.image_features: image_features,
                    model.answers: answer
                }
            )
            end = time.clock()
            print "Time for batch of photos", end - start
            print "Time for one epoch (mins)", len(qa_data['training'])/args.batch_size * (end - start)/60.0
            batch_no += 1
            step += 1
            
            print "LOSS", loss_value, batch_no,len(qa_data)/args.batch_size, step, epoch
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
                # just a sample
                actual_ans_text = utils.answer_indices_to_text(answer[:,0], ans_vocab_rev)
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
                gc.collect()

            if step % args.evaluate_every == 0:
                accuracy = evaluate_model(model, qa_data, args, 
                    model_options, sess, conv_features_val, image_id_map_val)
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
                gc.collect()
                # to avoid h5py from slowing down.
                conv_features = data_loader.load_conv_features('train', args.cnn_model, args.feature_layer, 
                    load_image_list = False)

            if step >= args.max_steps:
                break
        
def evaluate_model(model, qa_data, args, model_options, sess, conv_features, image_id_map):
    # to avoid h5py from slowing down.
    conv_features = data_loader.load_conv_features('val', args.cnn_model, args.feature_layer, 
        load_image_list = False)
    
    prediction_check = []
    ans_vocab_rev = qa_data['index_to_ans']  
    
    print "loading conv feats"
    
    batch_no = 0
    while (batch_no*args.batch_size) < len(qa_data['validation']):
        question, answer, image_features, image_ids, ans_freq_batch = get_batch(
                batch_no, args.batch_size, qa_data['validation'], 
                conv_features, image_id_map, 'val', model_options
                )
        [predicted] = sess.run([model.g_predictions], feed_dict = {
            model.g_question : question,
            model.g_image_features : image_features
        })
        pred_ans_text = utils.answer_indices_to_text(predicted, ans_vocab_rev)
        for bi, pred_ans in enumerate(pred_ans_text):
            if pred_ans in ans_freq_batch[bi]:# and ans_freq_batch[bi][pred_ans] >= 3:
                prediction_check.append(min(1.0, ans_freq_batch[bi][pred_ans]/3.0))
                # prediction_check.append(1.0)
            else:
                prediction_check.append(0.0)
            # print pred_ans, ans_freq_batch, prediction_check[-1]
        accuracy = np.sum(prediction_check, dtype = "float32")/len(prediction_check)
        print "Eavluating", batch_no, len(qa_data)/args.batch_size, accuracy
        batch_no += 1
        
    return accuracy


def get_batch(batch_no, batch_size, 
   qa, conv_features, image_id_map,
   split, model_options):
    img_dim = model_options['img_dim']
    img_channels = model_options['img_channels']

    si = (batch_no * batch_size)%len(qa)
    ei = min(len(qa), si + batch_size)
    n = ei - si
    max_question_length = model_options['max_question_length']

    sentence_batch = np.ndarray( (n, max_question_length), dtype = 'int32')
    answer_batch = np.zeros((n, model_options['num_answers']), dtype = 'int32' )
    conv_batch = np.ndarray((n, img_dim, img_dim, img_channels), dtype = 'float32')
    image_ids = []
    
    ans_freq_batch = []
    count = 0
    for i in range(si, ei):
        sentence_batch[count] = qa[i]['question_indices']
        answer_batch[count] = qa[i]['all_answers_indices']
        conv_index = image_id_map[ qa[i]['image_id'] ]
        conv_batch[count] = np.reshape(conv_features[conv_index], [14, 14, 2048] )
        image_ids.append(qa[i]['image_id']  )
        ans_freq_batch.append(qa[i]['ans_freq'])
        count += 1

    return sentence_batch, answer_batch, conv_batch, image_ids, ans_freq_batch

if __name__ == '__main__':
    main()