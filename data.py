import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import Counter, OrderedDict
from copy import deepcopy
flatten = lambda l: [item for sublist in l for item in sublist]
import json
import nltk
import os

THIS_PATH = os.path.dirname(os.path.abspath(__file__))
USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

def tokenize(sequence):
    tokens = [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(sequence)]
    return [x for x in tokens] # .encode('utf8')

def token_idx_map(context, context_tokens):
    acc = ''
    current_token_idx = 0
    token_map = dict()
    for char_idx, char in enumerate(context):
        if char != ' ':
            acc += char
            context_token = str(context_tokens[current_token_idx])
            if acc == context_token:
                syn_start = char_idx - len(acc) + 1
                token_map[syn_start] = [acc, current_token_idx]
                acc = ''
                current_token_idx += 1
    return token_map

    
def load_squad_data(data_path,max_len=600):
    dataset = json.load(open(data_path,'r'))
    data_p=[]
    qn, an = 0, 0
    skipped = 0
    
    for articles_id in range(len(dataset['data'])):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']
            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)
            if len(context_tokens)>max_len: continue

            answer_map = token_idx_map(context, context_tokens)

            qas = article_paragraphs[pid]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = tokenize(question)

                answers = qas[qid]['answers']
                qn += 1

                num_answers = list(range(1))

                for ans_id in num_answers:
                    # it contains answer_start, text
                    text = qas[qid]['answers'][ans_id]['text']
                    a_s = qas[qid]['answers'][ans_id]['answer_start']

                    text_tokens = tokenize(text)

                    answer_start = qas[qid]['answers'][ans_id]['answer_start']

                    answer_end = answer_start + len(text)

                    last_word_answer = len(text_tokens[-1]) # add one to get the first char

                    try:
                        a_start_idx = answer_map[answer_start][1]

                        a_end_idx = answer_map[answer_end - last_word_answer][1]

                        data_p.append([context_tokens,question_tokens,text_tokens,a_start_idx,a_end_idx])


                    except Exception as e:
                        skipped += 1

                    an += 1
    
    print("Skipped {}, {} question/answer".format(skipped, len(data_p)))
    
    return data_p
    

def prepare_sequence(seq, to_index):
    idxs = list(map(lambda w: to_index[w] if w in to_index.keys() else to_index["<unk>"], seq))
    return Variable(LongTensor(idxs))

def getBatch(batch_size,train_data):
    random.shuffle(train_data)
    sindex=0
    eindex=batch_size
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]
        temp = eindex
        eindex = eindex+batch_size
        sindex = temp
        yield batch
    
    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch
    
def pad_to_batch(batch,w_to_ix): # for Squad dataset
    doc,q,s,e = list(zip(*batch))
    max_doc = max([p.size(1) for p in doc])
    max_q = max([qq.size(1) for qq in q])
    
    doc_p,q_p = [],[]
    for i in range(len(batch)):
        
        if doc[i].size(1)<max_doc:
            doc_p.append(torch.cat([doc[i],Variable(LongTensor([w_to_ix['<pad>']]*(max_doc-doc[i].size(1)))).view(1,-1)],1))
        else:
            doc_p.append(doc[i])
        
        if q[i].size(1)<max_q:
            q_p.append(torch.cat([q[i],Variable(LongTensor([w_to_ix['<pad>']]*(max_q-q[i].size(1)))).view(1,-1)],1))
        else:
            q_p.append(q[i])

    docs  = torch.cat(doc_p)
    questions = torch.cat(q_p)
    starts = torch.cat(s)
    ends = torch.cat(e)
    return docs,questions,starts,ends
    
def preprop(dataset,word2index=None):
    docs,qu,_,start,end = list(zip(*dataset))
    
    if word2index is None:
        word2index={'<pad>':0,'<unk>':1,'<s>':2,'</s>':3}

        for tk in flatten(docs)+flatten(qu):
            if tk not in word2index.keys():
                word2index[tk]=len(word2index)

    print("Successfully Build %d vocabs" % len(word2index))
    
    data_p=[]
    for i in range(len(docs)):
        temp=[]
        temp.append(prepare_sequence(docs[i],word2index).unsqueeze(0))
        temp.append(prepare_sequence(qu[i],word2index).unsqueeze(0))
        temp.append(Variable(LongTensor([start[i]])).unsqueeze(0))
        temp.append(Variable(LongTensor([end[i]])).unsqueeze(0))
        data_p.append(temp)
    print("Preprop Complete!")
    
    return word2index,data_p