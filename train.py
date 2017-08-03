import tensorflow as tf
from Models import VQA_model
import data_loader
import argparse
import numpy as np
from os.path import isfile, join
import utils
import scipy.misc
import gc
import time

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--fc7_feature_length', type=int, default=4096,
                       help='fc7_feature_length')
    parser.add_argument('--residual_channels', type=int, default=512,
                       help='residual_channels')  
    parser.add_argument('--data_dir', type=str, default='Data',
                       help='Data directory')
    parser.add_argument('--batch_size', type=int, default=32,
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

    args = parser.parse_args()
    
    print "Reading QA DATA", args.version
    qa_data = data_loader.load_questions_answers(args.version, args.data_dir)
    qa_data['ans_vocab_rev'] = { qa_data['answer_vocab'][ans] : ans for ans in qa_data['answer_vocab']}
    qa_data['ques_vocab_rev'] = { qa_data['question_vocab'][qw] : qw for qw in qa_data['question_vocab']}
    qa_data['ques_vocab_rev'][0] = ""
    ans_vocab_rev = qa_data['ans_vocab_rev']

    model_options = {
        'residual_channels' : args.residual_channels,
        'text_length' : qa_data['max_question_length'],
        'ans_vocab_size' : len(qa_data['answer_vocab']),
        'encoder_filter_width' : 3,
        'batch_size' : args.batch_size,
        'words_vectors_provided' : True,
        'length_of_word_vector' : 300,
        'encoder_dilations' : [1, 2, 4, 8, 16],
        'img_dim' : 448
    }
    
    print "MODEL OPTIONS"
    print model_options
    model = VQA_model.VQA_model(model_options)
    # input_tensors, probability_maps, t_loss, t_accuracy, t_p, variables = model.build_model()
    vqa_model = model.build_model_attention()

    input_tensors = vqa_model['input_tensors']
    probability_maps = vqa_model['probability_maps']
    t_loss = vqa_model['loss']
    t_accuracy = vqa_model['accuracy']
    t_p = vqa_model['predictions']
    variables = vqa_model['var_list']
    vgg = vqa_model['vgg']

    pm1 = probability_maps['map1']
    pm2 = probability_maps['map2']
    
    train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(t_loss, var_list = variables['model_variables'])
    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()

    vgg.load_weights('Data/CNNModels/vgg16_weights.npz', sess)
    print "VGG WEIGHTS LOADED"

    saver = tf.train.Saver()
    if args.resume_model:
        saver.restore(sess, args.resume_model)

    gc.collect()
    for i in xrange(args.epochs):
        batch_no = 0

        while ((batch_no + 1)*args.batch_size) < len(qa_data['training']):
            start = time.clock()
            sentence, answer, image, sentence_words, answer_words = get_training_batch(batch_no, args.batch_size, qa_data, 'train')
            _, loss_value, accuracy, pred, pmap1, pmap2 = sess.run([train_op, t_loss, t_accuracy, t_p, pm1, pm2], 
                feed_dict={
                    input_tensors['image']:image,
                    input_tensors['source_sentence']:sentence,
                    input_tensors['answer']:answer
                }
            )
            end = time.clock()
            print "Time for batch 10 photos", end - start
            print "Time for one epoch", len(qa_data['training'])/args.batch_size * (end - start)
            batch_no += 1
            if args.debug:
                if batch_no % args.debug_every == 0:
                    save_batch(image, sentence_words, sentence, answer_words, pred, ans_vocab_rev, 'Data/debug')
                    for idx, p in enumerate(pred):
                        print ans_vocab_rev[p], ans_vocab_rev[ np.argmax(answer[idx])]

            print "Loss", loss_value, batch_no, i
            print "Accuracy", accuracy
            print "---------------"
        
        save_path = saver.save(sess, "Data/Models{}/model{}.ckpt".format(args.version, i))
        
def save_batch(image_batch, sentence_batch, sentence,
    answer_batch, predictions, ans_vocab_rev, data_dir):
    line_str = ""
    for i in range(len(sentence_batch)):
        sentence_text = " ".join(str(s) for s in sentence_batch[i])
        answer_text = answer_batch[i]
        line_str += "{} question = {} \n{} answer_actual = {} \n{} answer_pred = {} \n{} vector = {} \n".format(
            i, sentence_text, i, answer_text, i, ans_vocab_rev[ predictions[i]] , i, np.array_str(sentence[i]) )
        scipy.misc.imsave(join(data_dir, '{}.jpg'.format(i)), image_batch[i])

    with open(join(data_dir, "ques_ans.txt"), 'wb') as f:
        f.write(line_str)
        f.close()



def get_training_batch(batch_no, batch_size, 
    qa_data, split):
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
    image_batch = np.ndarray((n, 448, 448, 3), dtype = 'float32')

    # for displaying question and answers
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
        image_file = join('Data', '%s2014/COCO_%s2014_%.12d.jpg'%(split, split, qa[i]['image_id'] ) )
        image_array = utils.load_image_array(image_file)
        image_batch[count, :, :, :] = image_array
        count += 1

    return sentence_batch, answer_batch, image_batch, sentence_words_batch, answer_words_batch

if __name__ == '__main__':
    main()