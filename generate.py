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
from Models import resnet

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--residual_channels', type=int, default=512,
                       help='residual_channels')  
    parser.add_argument('--data_dir', type=str, default='Data',
                       help='Data directory')
    parser.add_argument('--version', type=int, default=1,
                       help='VQA data version')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Trained Model Path')
    parser.add_argument('--feature_layer', type=str, default="block4",
                       help='CONV FEATURE LAYER, fc7, pool5 or block4')
    parser.add_argument('--cnn_model', type=str, default="resnet",
                       help='CNN model')
    parser.add_argument('--text_model', type=str, default="bytenet",
                       help='bytenet/lstm')
    parser.add_argument('--question', type=str, default="What animal is shown in the picture",
                       help='question about the image')
    parser.add_argument('--image_file', type=str, default="Image File path for the question",
                       help='Image File path')

    
    args = parser.parse_args()
    conv_features_batch = get_conv_features(args.image_file, args.cnn_model, args.feature_layer)
    
    tf.reset_default_graph()

    meta_data = data_loader.load_meta_data(args.version, args.data_dir)
    ans_vocab_rev = meta_data['index_to_ans']
    ques_vocab_rev = meta_data['index_to_qw']
    qw_to_index = meta_data['qw_to_index']
    
    
    question_words = data_loader.tokenize_mcb(args.question)
    question_indices = [qw_to_index[qw] if qw in qw_to_index else qw_to_index['UNK'] 
    for qw in question_words]
    
    question_indices += [0 for i in range(len(question_indices), meta_data['max_question_length'])]
    sentence_batch = np.ndarray( (1, meta_data['max_question_length']), dtype = 'int32')
    sentence_batch[0] = question_indices

    

    model_options = {
        'question_vocab_size' : len(meta_data['index_to_qw']),
        'residual_channels' : args.residual_channels,
        'ans_vocab_size' : len(meta_data['index_to_ans']),
        'filter_width' : 3,
        'img_dim' : 14,
        'img_channels' : 2048,
        'dilations' : [ 1, 2, 4, 8,
                        1, 2, 4, 8, 
                       ],
        'text_model' : args.text_model,
        'dropout_keep_prob' : 0.6,
        'max_question_length' : meta_data['max_question_length'],
        'num_answers' : 10
    }
    
    
    model = VQA_model_attention.VQA_model(model_options)
    model.build_generator()

    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()

    saver = tf.train.Saver()
    if args.model_path:
        saver.restore(sess, args.model_path)


    try:
        shutil.rmtree('Data/gen_samples')
    except:
        pass
    
    os.makedirs('Data/gen_samples')

    pred_answer, prob1, prob2 = sess.run([model.g_predictions, model.g_prob1, model.g_prob2],
        feed_dict = {
            model.g_question : sentence_batch,
            model.g_image_features : conv_features_batch
        })

    pred_ans_text = utils.answer_indices_to_text(pred_answer, ans_vocab_rev)
    
    sample_data = []
    print "Actual vs Prediction"
    for sample_i in range(len(pred_ans_text)):
        print pred_ans_text[sample_i]
        image_array = utils.load_image_array(args.image_file, 224)
        blend1 = utils.get_blend_map(image_array, prob1[sample_i], overlap = True)
        blend2 = utils.get_blend_map(image_array, prob2[sample_i], overlap = True)
        sample_data.append({
            'question' : args.question,
            'predicted_answer' : pred_ans_text[sample_i],
            'batch_index' : sample_i
            })
        misc.imsave('Data/gen_samples/{}_actual_image.jpg'.format(sample_i), image_array)
        misc.imsave('Data/gen_samples/{}_blend1.jpg'.format(sample_i), blend1)
        misc.imsave('Data/gen_samples/{}_blend2.jpg'.format(sample_i), blend2)

        f = open('Data/gen_samples/sample.json', 'wb')
        f.write(json.dumps(sample_data))
        f.close()
        shutil.make_archive('Data/gen_samples', 'zip', 'Data/gen_samples')  
        
def get_conv_features(image_file, model_type, feature_layer):
    if model_type=="vgg":
        cnn_model = vgg16.create_vgg_model(448, only_conv = feature_layer != 'fc7')
    else:
        cnn_model = resnet.create_resnet_model(448)

    sess = cnn_model['session']
    images = cnn_model['images_placeholder']
    image_feature_layer = cnn_model[feature_layer]
    img_dim = 448

    if model_type == 'resnet':
        image_array = sess.run(cnn_model['processed_image'], feed_dict = {
            cnn_model['pre_image'] : utils.load_image_array(image_file, img_dim = None)
            })
    else:
        image_array = utils.load_image_array(image_file, img_dim = img_dim)

    feed_dict  = { images : [image_array] }
    conv_features_batch = sess.run(image_feature_layer, feed_dict = feed_dict)
    sess.close()

    return conv_features_batch

if __name__ == '__main__':
    main()