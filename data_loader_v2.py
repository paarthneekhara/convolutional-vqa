import json
import argparse
from os.path import isfile, join
import re
import numpy as np
import pprint
import h5py
from nltk.tokenize import word_tokenize

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

    combined_training_data = combine_question_answers(t_questions, t_answers)
    combined_val_data = combine_question_answers(v_questions, v_answers)

    index_to_ans, ans_to_index = get_top_answers(t_answers)
    frequent_training_data = remove_infrequent_examples(combined_training_data, ans_to_index)

    index_to_qw, qw_to_index = make_question_vocab(frequent_training_data)
    
    


def process_data(data_split):
    training_data_cleaned = []
    for eg in data_split:
        row = {}
        question_words = tokenize_mcb(eg['question_data']['question'])
        question_indices = [qw_to_index[qw] if qw in qw_to_index else qw_to_index['UNK'] 
        for qw in question_words]
        question_indices = question_indices[0:options['max_length']]
        question_indices += [0 for i in range(len(question_indices), options['max_length'])]
        row['question_indices'] : question_indices
        row['']
        answers =  

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

    question_words['UNK'] = True
    index_to_qw = [qw for qw in question_vocab]
    qw_to_index = {qw : i for i, qw in enumerate(index_to_qw)}

    print "Question Vocab Size", len(index_to_qw)
    return index_to_qw, qw_to_index



def combine_question_answers(questions, answers):
    comined = []
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
    for annotation in answers:
        answer = annotation['multiple_choice_answer']
        if answer in ans_freq:
            ans_freq[answer] += 1
        else:
            ans_freq[answer] = 1

    answer_frequency_tuples = [ (-freq, ans) for ans, freq in ans_freq]
    answer_frequency_tuples.sort()
    answer_frequency_tuples = answer_frequency_tuples[0:top_n-1]

    answer_vocab = {ans_tuple[1] : True for ans_tuple in answer_frequency_tuples}
    answer_vocab['UNK'] = True

    index_to_ans = [ans for ans in answer_vocab]
    ans_to_index = {ans:i for i, ans in enumerate(index_to_ans)}
    return index_to_ans, ans_to_index


