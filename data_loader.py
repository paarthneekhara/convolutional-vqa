import json
import argparse
from os.path import isfile, join
import re
import numpy as np
import pprint
import h5py
# from nltk.tokenize import word_tokenize

def prepare_training_data(version = 2, data_dir = 'Data', options = {}):
    if version == 1:
        t_q_json_file = join(data_dir, 'MultipleChoice_mscoco_train2014_questions.json')
        t_a_json_file = join(data_dir, 'mscoco_train2014_annotations.json')

        v_q_json_file = join(data_dir, 'MultipleChoice_mscoco_val2014_questions.json')
        v_a_json_file = join(data_dir, 'mscoco_val2014_annotations.json')
        qa_data_file = join(data_dir, 'qa_data_file1.json')
        vocab_file = join(data_dir, 'vocab_file1.json')
    else:
        t_q_json_file = join(data_dir, 'v2_OpenEnded_mscoco_train2014_questions.json')
        t_a_json_file = join(data_dir, 'v2_mscoco_train2014_annotations.json')

        v_q_json_file = join(data_dir, 'v2_OpenEnded_mscoco_val2014_questions.json')
        v_a_json_file = join(data_dir, 'v2_mscoco_val2014_annotations.json')
        qa_data_file = join(data_dir, 'qa_data_file2.json')
        vocab_file = join(data_dir, 'vocab_file2.json')

    print "Loading Training questions"
    with open(t_q_json_file) as f:
        t_questions = json.loads(f.read())
    
    print "Loading Training anwers"
    with open(t_a_json_file) as f:
        t_answers = json.loads(f.read())

    print "Loading Val questions"
    with open(v_q_json_file) as f:
        v_questions = json.loads(f.read())
    
    print "Loading Val answers"
    with open(v_a_json_file) as f:
        v_answers = json.loads(f.read())

    print "Total Training Questions", len(t_questions['questions']), len(t_answers['annotations'])
    print "Combining Training Data"
    combined_training_data = combine_question_answers(t_questions, t_answers)
    print "Combining Validation Data"
    combined_val_data = combine_question_answers(v_questions, v_answers)

    if options['train_on_val']:
        combined_training_data += combined_val_data

    index_to_ans, ans_to_index = get_top_answers(t_answers)
    print "Check ans vocab"
    print index_to_ans[0:5]
    for word in index_to_ans[0:5]:
        print word, ans_to_index[word]

    frequent_training_data = remove_infrequent_examples(combined_training_data, ans_to_index)
    
    index_to_qw, qw_to_index = make_question_vocab(frequent_training_data)
    
    print "Check question vocab"
    print index_to_qw[0:5]
    for word in index_to_qw[0:5]:
        print word, qw_to_index[word]

    training_data_processed = process_data(frequent_training_data, qw_to_index, ans_to_index, options)
    val_data_processed = process_data(combined_val_data, qw_to_index, ans_to_index, options)

    for row in training_data_processed[0:2]:
        pprint.pprint(row)

    print "++++++++++++++++++++++++++"
    for row in val_data_processed[0:2]:
        pprint.pprint(row)

    data = {
        'training' : training_data_processed,
        'validation' : val_data_processed,
        'index_to_qw' : index_to_qw,
        'qw_to_index' : qw_to_index,
        'index_to_ans' : index_to_ans,
        'ans_to_index' : ans_to_index,
        'max_question_length' : options['max_length']
    }
    
    meta_data = {
        'index_to_qw' : index_to_qw,
        'qw_to_index' : qw_to_index,
        'index_to_ans' : index_to_ans,
        'ans_to_index' : ans_to_index,
        'max_question_length' : options['max_length']
    }

    with open(qa_data_file, 'wb') as f:
        f.write(json.dumps(data))

    with open(vocab_file, 'wb') as f:
        f.write(json.dumps(meta_data))

def process_data(data_split, qw_to_index, ans_to_index, options):
    data_processed = []
    for eg in data_split:
        row = {}
        question_words = tokenize_mcb(eg['question_data']['question'])
        question_indices = [qw_to_index[qw] if qw in qw_to_index else qw_to_index['UNK'] 
        for qw in question_words]
        question_indices = question_indices[0:options['max_length']]
        question_indices += [0 for i in range(len(question_indices), options['max_length'])]

        all_answers = [ ans['answer'] for ans in eg['answer_data']['answers'] ]
        all_answers_indices = [ ans_to_index[ans] if ans in ans_to_index else ans_to_index['UNK'] 
        for ans in  all_answers]

        ans_freq = {}
        for ans in all_answers:
            if ans in ans_freq:
                ans_freq[ans] += 1
            else:
                ans_freq[ans] = 1

        row['image_id'] = eg['answer_data']['image_id']
        row['question_indices'] = question_indices
        row['all_answers_indices'] = all_answers_indices
        row['all_answers_words'] = all_answers
        row['ans_freq'] = ans_freq
        row['question_words'] = question_words
        row['best_ans_word'] = eg['answer_data']['multiple_choice_answer']
        if row['best_ans_word'] in ans_to_index:
            row['best_ans_index'] = ans_to_index[ row['best_ans_word'] ]
        else:
            row['best_ans_index'] = ans_to_index['UNK']

        # REPLACE UNK WITH MOST COMMON ANS
        for a_no in range(len(row['all_answers_indices'])):
            if row['all_answers_indices'][a_no] == ans_to_index['UNK']:
                row['all_answers_indices'][a_no] = row['best_ans_index']


        data_processed.append( row )
        # row['']
        # answers =  

    return data_processed

# borrowed MCB tokenize code
def tokenize_mcb(s):
    t_str = s.lower()
    for i in [r'\?',r'\!',r'\'',r'\"',r'\$',r'\:',r'\@',r'\(',r'\)',r'\,',r'\.',r'\;']:
        t_str = re.sub( i, '', t_str)
    for i in [r'\-',r'\/']:
        t_str = re.sub( i, ' ', t_str)
    q_list = re.sub(r'\?','',t_str.lower()).split(' ')
    q_list = list(filter(lambda x: len(x) > 0, q_list))
    return q_list

def make_question_vocab(questions):
    question_vocab = {}
    for question in questions:
        question_str = question['question_data']['question']
        question_words = tokenize_mcb(question_str)
        for qw in question_words:
            question_vocab[qw] = True

    question_vocab['UNK'] = True
    index_to_qw = [qw for qw in question_vocab]
    index_to_qw = ['#PAD#'] + index_to_qw

    qw_to_index = {qw : i for i, qw in enumerate(index_to_qw)}

    print "Question Vocab Size", len(index_to_qw)
    return index_to_qw, qw_to_index



def combine_question_answers(questions, answers):
    combined = []
    for i, question in enumerate(questions['questions']):
        combined.append({
            'question_data' : question,
            'answer_data' : answers['annotations'][i]
            })
    return combined

def remove_infrequent_examples(training_data, ans_to_index):
    frequent_data = []
    for eg in training_data:
        if eg['answer_data']['multiple_choice_answer'] in ans_to_index:
            frequent_data.append(eg)

    print "Total questions", len(training_data)
    print "Questions after removing less frequent ans", len(frequent_data)

    return frequent_data

def get_top_answers(answers, top_n = 3000):
    ans_freq = {}
    for annotation in answers['annotations']:
        answer = annotation['multiple_choice_answer']
        if answer in ans_freq:
            ans_freq[answer] += 1
        else:
            ans_freq[answer] = 1

    answer_frequency_tuples = [ (-freq, ans) for ans, freq in ans_freq.iteritems()]
    answer_frequency_tuples.sort()
    answer_frequency_tuples = answer_frequency_tuples[0:top_n-1]

    answer_vocab = {ans_tuple[1] : True for ans_tuple in answer_frequency_tuples}
    answer_vocab['UNK'] = True

    index_to_ans = [ans for ans in answer_vocab]
    ans_to_index = {ans:i for i, ans in enumerate(index_to_ans)}
    return index_to_ans, ans_to_index


def load_questions_answers(version = 1, data_dir = 'Data'):
    qa_data_file = join(data_dir, 'qa_data_file{}.json'.format(version))
    with open(qa_data_file) as f:
        data = json.loads(f.read())
        return data

def load_meta_data(version = 1, data_dir = 'Data'):
    meta_data = join(data_dir, 'vocab_file{}.json'.format(version))
    with open(meta_data) as f:
        data = json.loads(f.read())
        return data

def load_conv_features(split = 'train', model = "resnet", feature_layer = 'block4', 
    load_image_list = True):
    conv_file = "Data/conv_features_{}_{}/conv_features_{}.h5".format(split, model, feature_layer)
    # conv_file = "Data/conv_features_{}/conv_features_{}_bucket_{}.h5".format(split, feature_layer, bucket_no)
    hf = h5py.File( conv_file,'r')
    conv_features = hf['conv_features']
    print "Shape", conv_features.shape
    if not load_image_list:
        return conv_features
    # print conv_features[1,:,:,:]
    image_id_file = "Data/conv_features_{}_{}/image_id_list_{}.h5".format(split, model, feature_layer)
    # image_id_file = "Data/conv_features_{}/image_id_list_bucket_{}.h5".format(split, bucket_no)
    with h5py.File( image_id_file,'r') as hf:
        image_id_list = np.array(hf.get('image_id_list'))

    return conv_features, image_id_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, default=2,
                       help='VQA data version')
    parser.add_argument('--max_length', type=int, default=15,
                       help='VQA data version')
    parser.add_argument('--train_on_val', type=bool, default=False,
                       help='VQA data version')
    args = parser.parse_args()
    options = {
        'max_length' : args.max_length,
        'train_on_val' : args.train_on_val
    }

    prepare_training_data(args.version, options = options)

if __name__ =='__main__':
    main()
