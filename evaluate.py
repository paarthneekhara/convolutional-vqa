import tensorflow as tf
from Models import VQA_model_fc7
import data_loader
import argparse
import numpy as np

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
    parser.add_argument('--epochs', type=int, default=200,
                       help='Expochs')
    parser.add_argument('--version', type=int, default=2,
                       help='VQA data version')
    parser.add_argument('--debug', type=bool, default=False,
                       help='Debug')
    parser.add_argument('--model_path', type=str, default="Data/Models1_lstm_fc7/model4.ckpt",
                       help='Trained Model Path')
    parser.add_argument('--text_model', type=str, default='bytenet',
                       help='Text model to choose : bytenet/lstm')

    args = parser.parse_args()
    print "Reading QA DATA"
    qa_data = data_loader.load_questions_answers(args.version, args.data_dir)
    
    print "Reading fc7 features"
    fc7_features, image_id_list = data_loader.load_fc7_features(args.data_dir, 'vgg', args.version, 'val')
    print "FC7 features", fc7_features.shape
    print "image_id_list", image_id_list.shape

    image_id_map = {}
    for i in xrange(len(image_id_list)):
        image_id_map[ image_id_list[i] ] = i

    ans_map = { qa_data['answer_vocab'][ans] : ans for ans in qa_data['answer_vocab']}

    model_options = {
        'residual_channels' : args.residual_channels,
        'fc7_feature_length' : args.fc7_feature_length,
        'text_length' : qa_data['max_question_length'],
        'n_source_quant' : len(qa_data['question_vocab']),
        'ans_vocab_size' : len(qa_data['answer_vocab']),
        'encoder_filter_width' : 3,
        'batch_size' : args.batch_size,
        'encoder_dilations' : [1, 2, 4, 8, 16,
                          1, 2, 4, 8, 16],
        'length_of_word_vector' : 300,
        'num_lstm_layers' : 2,
        'img_dim' : 224,
        'text_model' : args.text_model
    }
    
    
    
    model = VQA_model_fc7.VQA_model(model_options)
    model = VQA_model_fc7.VQA_model(model_options)
    
    vqa_model = model.build_model(train = False)
    input_tensors = vqa_model['input_tensors']
    t_accuracy = vqa_model['accuracy']
    t_p = vqa_model['predictions']

    sess = tf.InteractiveSession()
    # tf.initialize_all_variables().run()
    # saver = tf.train.import_meta_graph('Data/Models/model20.ckpt.meta')
    saver = tf.train.Saver()
    saver.restore(sess,args.model_path)


    avg_accuracy = 0.0
    total = 0
    # saver.restore(sess, args.model_path)
    
    batch_no = 0
    while ((batch_no+1)*args.batch_size) < len(qa_data['validation']):
        sentence, answer, fc7 = get_batch(batch_no, args.batch_size, 
            fc7_features, image_id_map, qa_data, 'val')
        
        pred = sess.run([t_p], feed_dict={
            input_tensors['fc7']:fc7,
            input_tensors['source_sentence']:sentence,
            input_tensors['answer']:answer
        })
        
        batch_no += 1
        if args.debug:
            for idx, p in enumerate(pred):
                print ans_map[p], ans_map[ np.argmax(answer[idx])]

        correct_predictions = np.equal(pred, np.argmax(answer, 1))
        correct_predictions = correct_predictions.astype('float32')
        accuracy = correct_predictions.mean()
        print "Acc", accuracy
        avg_accuracy += accuracy
        total += 1
    
    print "Acc", avg_accuracy/total


def get_batch(batch_no, batch_size, fc7_features, image_id_map, qa_data, split):
    qa = None
    if split == 'train':
        qa = qa_data['training']
    else:
        qa = qa_data['validation']

    max_question_length = qa_data['max_question_length']
    si = (batch_no * batch_size)%len(qa)
    ei = min(len(qa), si + batch_size)
    n = ei - si
    sentence_batch = np.ndarray( (n, max_question_length, 300), dtype = 'float32')
    answer = np.zeros( (n, len(qa_data['answer_vocab'])))
    fc7 = np.ndarray( (n,4096) )
    word_vectors = qa_data['word_vectors']
    word_vectors[0] = np.zeros(300)
    count = 0

    for i in range(si, ei):
        for qwi in xrange(max_question_length):
            sentence_batch[count,qwi,:] =  word_vectors[ qa[i]['question'][qwi] ]
        answer[count, qa[i]['answer']] = 1.0
        fc7_index = image_id_map[ qa[i]['image_id'] ]
        fc7[count,:] = fc7_features[fc7_index][:]
        count += 1
    
    return sentence_batch, answer, fc7

if __name__ == '__main__':
    main()