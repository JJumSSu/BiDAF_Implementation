import json
import os
import torch
import logging
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
from util import word_tokenize
import pdb

# Static?

logging.getLogger().setLevel(logging.INFO)

class READ():

    def __init__(self, args):

        path = '../data/squad'

        logging.info("Preprocessing Data - First Phase  :: Reading And Transforming")

        self.preprocess('{}/{}'.format(path, args.Train_File))
        self.preprocess('{}/{}'.format(path, args.Dev_File))

        self.RAW = data.RawField(); self.RAW.is_target = False

        self.CHAR_NESTING  = data.Field(batch_first = True, tokenize = list, lower=True)
        self.CHAR  = data.NestedField(self.CHAR_NESTING, tokenize = word_tokenize)
        self.WORD  = data.Field(batch_first = True, tokenize = word_tokenize, lower = True, include_lengths = True)
        self.LABEL = data.Field(sequential = False, unk_token = None, use_vocab = False)

        dict_fields = {'qid'      : ('qid', self.RAW),
                       'start_idx': ('start_idx', self.LABEL),
                       'end_idx'  : ('end_idx', self.LABEL),
                       'context'  : [('c_word', self.WORD), ('c_char', self.CHAR)],
                       'question' : [('q_word', self.WORD), ('q_char', self.CHAR)]}
        
        logging.info("Preprocessing Data - Second Phase :: To Torchtext")
        
        self.train, self.dev = data.TabularDataset.splits(path=path, train=args.Train_File + 'l',  \
                                                          validation=args.Dev_File + 'l', format='json', fields=dict_fields)
        if args.Max_Token_Length > 0:
            self.train.examples = [e for e in self.train.examples if len(e.c_word) <= args.Max_Token_Length]

        logging.info("Preprocessing Data - Third Phase  :: Building Vocabulary")
        
        self.CHAR.build_vocab(self.train, self.dev)
        self.WORD.build_vocab(self.train, self.dev, vectors=GloVe(name='6B', dim=args.Word_Dim))

        logging.info("Preprocessing Data - Fourth Phase :: Building Itertors")

        device = torch.device("cuda:{}".format(args.GPU) if torch.cuda.is_available() else "cpu")
        
        self.train_iter = data.BucketIterator(dataset = self.train, batch_size = args.Batch_Size) # sort_key = lambda x : len(x.c_word)
        
        self.dev_iter   = data.BucketIterator(dataset = self.dev, batch_size = 10)


    def preprocess(self, path):

        output = []
        stopwords = [' ', '\n', '\u3000', '\u202f', '\u2009']

        with open(path, 'r', encoding='utf-8') as f:
            
            with open(path, 'r', encoding = 'utf-8') as t:
                data = []
                for line in t:
                    data.append(json.loads(line))
                t.close()

            for topic in data[0]['data']:
                for paragraph in topic['paragraphs']:
                    context = paragraph['context']
                    tokens = word_tokenize(context)
                    for qa in paragraph['qas']:
                        qid = qa['id']
                        question = qa['question']
                        for ans in qa['answers']:
                            answer = ans['text']
                            s_idx = ans['answer_start']
                            e_idx = s_idx + len(answer)

                            l = 0
                            s_found = False
                            for i, t in enumerate(tokens):
                                while l < len(context):
                                    if context[l] in stopwords:
                                        l += 1
                                    else:
                                        break
                                if t[0] == '"' and context[l:l + 2] == '\'\'':
                                    t = '\'\'' + t[1:]
                                elif t == '"' and context[l:l + 2] == '\'\'':
                                    t = '\'\''

                                l += len(t)
                                if l > s_idx and s_found == False:
                                    s_idx = i
                                    s_found = True
                                if l >= e_idx:
                                    e_idx = i
                                    break

                            output.append(dict([('qid', qid),
                                                ('context', context),
                                                ('question', question),
                                                ('answer', answer),
                                                ('start_idx', s_idx),
                                                ('end_idx', e_idx)]))

        with open('{}l'.format(path), 'w', encoding='utf-8') as f:
            for line in output:
                json.dump(line, f)
                print('', file=f)

