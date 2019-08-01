import logging
import os
import copy
import json 

import torch
from torch import nn, optim

from prepro import READ
from model import BiDAF, EMA
from util import evaluate

logging.getLogger().setLevel(logging.INFO)

class SOLVER():

    def __init__(self, args):
        self.args    =  args
        self.device  =  torch.device("cuda:{}".format(self.args.GPU) if torch.cuda.is_available() else "cpu")
        self.data    =  READ(self.args)
        glove        =  self.data.WORD.vocab.vectors
        char_size    =  len(self.data.CHAR.vocab)

        self.model   =  BiDAF(self.args, char_size, glove).to(self.device)
        self.ema     =  EMA(self.args.Exp_Decay_Rate)

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.ema.register(name, param.data)      
        
        self.parameters = filter(lambda p: p.requires_grad, self.model.parameters())

    def train(self):

        optimizer = optim.Adadelta(self.parameters, lr=self.args.Learning_Rate)
        criterion = nn.CrossEntropyLoss()

        self.model.train()

        max_dev_em, max_dev_f1 = -1, -1
        num_batches = len(self.data.train_iter)

        logging.info("Begin Training")
        
        self.model.zero_grad()

        loss = 0.0

        for epoch in range(self.args.Epoch):    
            for i, batch in enumerate(self.data.train_iter):
                p1, p2 = self.model(batch)
                batch_loss = criterion(p1, batch.start_idx) + criterion(p2, batch.end_idx)
                loss = float(batch_loss)
                batch_loss.backward()
                optimizer.step()
                del p1, p2, batch_loss

                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        self.ema.update(name, param.data)

                self.model.zero_grad()

                logging.info("Epoch [{}/{}] Step [{}/{}] Train Loss {}".format(epoch+1, self.args.Epoch, \
                                                                               i+1, int(num_batches) +1, round(loss,3)))
                if (i+1) % 700 == 0:
                    dev_em, dev_f1 = self.evaluate()
                    logging.info("Epoch [{}/{}] Dev EM {} Dev F1 {}".format(epoch + 1, self.args.Epoch, \
                                                                            round(dev_em,3), round(dev_f1,3)))
                    self.model.train()

            if dev_f1 > max_dev_f1:
                max_dev_f1 = dev_f1
                max_dev_em = dev_em
                # best_model = copy.deepcopy(self.model)
            

        # if not os.path.exists('../models'):
        #     os.mkdir("../models")
        #     torch.save(best_model, "../models/best_model.pt")

        logging.info('Max Dev EM: {} Max Dev F1: {}'.format(round(max_dev_em, 3), round(max_dev_f1, 3)))


    def evaluate(self): 

        logging.info("Evaluating on Dev Dataset")
        answers = dict()

        self.model.eval()

        temp_ema = EMA(0)

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                temp_ema.register(name, param.data)
                param.data.copy_(self.ema.get(name))
    
        with torch.set_grad_enabled(False):
            for _ , batch in enumerate(self.data.dev_iter):

                p1, p2 = self.model(batch)
                batch_size, _ = p1.size()

                _ , s_idx = p1.max(dim=1)
                _ , e_idx = p2.max(dim=1)

                for i in range(batch_size):
                    qid = batch.qid[i]
                    answer = batch.c_word[0][i][s_idx[i]:e_idx[i]+1]
                    answer = ' '.join([self.data.WORD.vocab.itos[idx] for idx in answer])
                    answers[qid] = answer                

            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data.copy_(temp_ema.get(name))

        results = evaluate(self.args, answers)

        return results['exact_match'], results['f1']



