import json
import argparse
from os.path import isfile, join
import re
import numpy as np
import pprint
import h5py

def prepare_training_data(version = 2, data_dir = 'Data'):
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

    # IF ALREADY EXTRACTED
    # qa_data_file = join(data_dir, 'qa_data_file{}.pkl'.format(version))
    
    # if isfile(qa_data_file):
    #     with open(qa_data_file) as f:
    #         data = pickle.load(f)
    #         return data

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

    
    print "Ans", len(t_answers['annotations']), len(v_answers['annotations'])
    print "Qu", len(t_questions['questions']), len(v_questions['questions'])

    answers = t_answers['annotations'] + v_answers['annotations']
    questions = t_questions['questions'] + v_questions['questions']
    
    answer_vocab = make_answer_vocab(answers)
    question_vocab, max_question_length = make_questions_vocab(questions, answers, answer_vocab)
    print "Max Question Length", max_question_length
    print "Question Vocab Length", len(question_vocab)
    print len(question_vocab)
    word_regex = re.compile(r'\w+')
    training_data = []
    for i,question in enumerate( t_questions['questions']):
        best_ans = t_answers['annotations'][i]['multiple_choice_answer']
        all_answers = [ ans['answer'] for ans in t_answers['annotations'][i]['answers'] ]
        all_answers_indices = [ answer_vocab[ans] if ans in answer_vocab else answer_vocab['UNK'] for ans in  all_answers]
        if ans in answer_vocab:
            training_data.append({
                'image_id' : t_answers['annotations'][i]['image_id'],
                'question' : [0 for i in range(max_question_length)],
                'answer' : best_ans,
                'all_answers' : all_answers_indices
                })
            question_words = re.findall(word_regex, question['question'])
            question_words = [qw.lower() for qw in question_words]
            for i in range(0, len(question_words)):
                training_data[-1]['question'][i] = question_vocab[ question_words[i] ]

    print "Training Data", len(training_data)
    val_data = []
    
    for i,question in enumerate( v_questions['questions']):
        ans = v_answers['annotations'][i]['multiple_choice_answer']
        all_answers = [ ans['answer'] for ans in v_answers['annotations'][i]['answers'] ]
        all_answers_indices = [ answer_vocab[ans] if ans in answer_vocab else answer_vocab['UNK'] for ans in  all_answers]
        ans_freq = {}
        for ans in all_answers:
            if ans in ans_freq:
                ans_freq[ans] += 1
            else:
                ans_freq[ans] = 1

        most_frequent_ans = answer_vocab[ans] if ans in answer_vocab else answer_vocab['UNK']
        val_data.append({
            'image_id' : v_answers['annotations'][i]['image_id'],
            'question' : [0 for i in range(max_question_length)],
            'answer' : most_frequent_ans,
            'all_answers_frequency' : ans_freq,
            'all_answers' : all_answers_indices
            })
        question_words = re.findall(word_regex, question['question'])
        question_words = [qw.lower() for qw in question_words]
        for i in range(0, len(question_words)):
            val_data[-1]['question'][i] = question_vocab[ question_words[i] ]

    print "Validation Data", len(val_data)

    data = {
        'training' : training_data,
        'validation' : val_data,
        'answer_vocab' : answer_vocab,
        'question_vocab' : question_vocab,
        'max_question_length' : max_question_length
    }

    print "Saving qa_data"
    
    with open(qa_data_file, 'wb') as f:
        f.write(json.dumps(data))

    with open(vocab_file, 'wb') as f:
        vocab_data = {
            'answer_vocab' : data['answer_vocab'],
            'question_vocab' : data['question_vocab'],
            'max_question_length' : data['max_question_length']
        }
        f.write(json.dumps(vocab_data))

    return data
    
def load_questions_answers(version = 2, data_dir = 'Data'):
    qa_data_file = join(data_dir, 'qa_data_file{}.json'.format(version))
    
    if isfile(qa_data_file):
        with open(qa_data_file) as f:
            data = json.loads(f.read())
            return data

def get_question_answer_vocab(version = 2, data_dir = 'Data'):
    vocab_file = join(data_dir, 'vocab_file{}.pkl'.format(version))
    vocab_data = json.loads(open(vocab_file).read())
    return vocab_data

def make_answer_vocab(answers):
    top_n = 3000
    answer_frequency = {} 
    for annotation in answers:
        answer = annotation['multiple_choice_answer']
        if answer in answer_frequency:
            answer_frequency[answer] += 1
        else:
            answer_frequency[answer] = 1

    answer_frequency_tuples = [ (-frequency, answer) for answer, frequency in answer_frequency.iteritems()]
    answer_frequency_tuples.sort()
    answer_frequency_tuples = answer_frequency_tuples[0:top_n-1]

    answer_vocab = {}
    for i, ans_freq in enumerate(answer_frequency_tuples):
        # print i, ans_freq
        ans = ans_freq[1]
        answer_vocab[ans] = i

    answer_vocab['UNK'] = top_n - 1
    return answer_vocab


def make_questions_vocab(questions, answers, answer_vocab):
    word_regex = re.compile(r'\w+')
    question_frequency = {}

    max_question_length = 0
    for i,question in enumerate(questions):
        count = 0
        question_words = re.findall(word_regex, question['question'])
        question_words = [qw.lower() for qw in question_words]
        for qw in question_words:
            if qw in question_frequency:
                question_frequency[qw] += 1
            else:
                question_frequency[qw] = 1
            count += 1
        if count > max_question_length:
            max_question_length = count


    qw_freq_threhold = 0
    qw_tuples = [ (-frequency, qw) for qw, frequency in question_frequency.iteritems()]
    # qw_tuples.sort()

    qw_vocab = {}
    for i, qw_freq in enumerate(qw_tuples):
        frequency = -qw_freq[0]
        qw = qw_freq[1]
        # print frequency, qw
        if frequency > qw_freq_threhold:
            # +1 for accounting the zero padding for batc training
            qw_vocab[qw] = i + 1
        else:
            break

    qw_vocab['UNK'] = len(qw_vocab) + 1

    return qw_vocab, max_question_length

def load_conv_features(split = 'train', bucket_no = 0, feature_layer = 'block4'):
    conv_file = "Data/conv_features_{}/conv_features_{}_bucket_{}.h5".format(split, feature_layer, bucket_no)
    hf = h5py.File( conv_file,'r')
    conv_features = hf['conv_features']
    print "Shape", conv_features.shape
    # print conv_features[1,:,:,:]
    image_id_file = "Data/conv_features_{}/image_id_list_bucket_{}.h5".format(split, bucket_no)
    with h5py.File( image_id_file,'r') as hf:
        image_id_list = np.array(hf.get('image_id_list'))

    return conv_features, image_id_list

def main():
    print "wtfh"
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, default=2,
                       help='VQA data version')
    args = parser.parse_args()
    prepare_training_data(args.version)

if __name__ =='__main__':
    main()