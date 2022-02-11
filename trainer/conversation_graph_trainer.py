import logging
from copy import deepcopy

import os
import math
import pandas as pd
import numpy as np
import torch
import pickle
from tqdm import tqdm
from model import *
import torch.nn as nn
import torch.optim as optim
import shutil
import tensorboardX
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy import stats
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


"""
Three sections: presentation, question, answers
Considering the consistency of all benchmarks, we use the sentence transformer as a substitute for Bi-lstm+attention

Todo List:
-[ ] check whether the dataset has been generated, if not, generate it.
"""


class ConversationGraphDataset(Dataset):
    def __init__(self, paths, label_name='firm_std_10_post'):
        super(ConversationGraphDataset, self).__init__()
        self.label_name = label_name
        data = self.loading_train_dataset(paths, label_name)
        self.input_pre = data['pre']
        self.input_pre_num = data['pre_num']
        self.input_qa = data['qa']
        self.input_qa_num = data['qa_num']
        self.input_rule_mask = data['rule_mask']
        self.input_affinity_matrix = data['affinity_matrix']
        self.label = np.log(data['label'])


    '''
    We simplify some details (mask operation) here so that the readers can easily understand what we are doing and 
    apply to their own specific tasks.
    
    In concrete, we drop the collect_fn function, which we design for masking different length's data, 
    and pre_num, q_num and a_num are use to generate mask
    '''

    @staticmethod
    def loading_train_dataset(paths: list, label_name: str) -> dict:
        X_pre = []
        X_qa = []

        X_pre_num = []
        X_qa_num = []

        X_rule = []
        X_affinity = []

        Y = []
        for path in paths:
            with open(path, 'rb') as fIn:
                stored_datas = pickle.load(fIn)
                for stored_data in tqdm(stored_datas):  # call
                    if label_name == 'firm_std_3_post':
                        Y.append(stored_data['label']['firm_std_3_post'])
                    elif label_name == 'firm_std_7_post':
                        Y.append(stored_data['label']['firm_std_7_post'])
                    elif label_name == 'firm_std_10_post':
                        Y.append(stored_data['label']['firm_std_10_post'])
                    elif label_name == 'firm_std_15_post':
                        Y.append(stored_data['label']['firm_std_15_post'])
                    elif label_name == 'firm_std_20_post':
                        Y.append(stored_data['label']['firm_std_20_post'])
                    elif label_name == 'firm_std_60_post':
                        Y.append(stored_data['label']['firm_std_60_post'])
                    else:
                        continue

                    X_pre.append(stored_data['pre_reps'])
                    X_pre_num.append(stored_data['pre_num'])
                    X_qa.append(stored_data['qa_reps'])
                    X_qa_num.append(stored_data['qa_num'])
                    X_rule.append(stored_data['rule_mask'])
                    X_affinity.append(stored_data['affinity_matrix'])

        return {
            'pre': X_pre,  # [M, N^p, d]
            'qa': X_qa,  # [M, N^{qa}, d]
            'pre_num': X_pre_num,
            'qa_num': X_qa_num,
            'rule_mask': X_rule,  # [M, N, d]
            'affinity_matrix': X_affinity,  # [M, N, d]
            'label': Y  # [M]
        }

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return {
            'pre': self.input_pre[index],
            'qa': self.input_qa[index],
            # 'pre_num': self.input_pre_num[index],
            # 'qa_num': self.input_qa_num[index],
            'rule_mask': self.input_rule_mask[index],
            'affinity_matrix': self.input_affinity_matrix[index],
            'label': self.label[index]
        }


class ModelWrappedWithMSELoss(nn.Module):
    def __init__(self,
                 device
                 ):
        super(ModelWrappedWithMSELoss, self).__init__()
        self.model = None
        self.criterion_risk = torch.nn.MSELoss(reduction='none')
        self.device = device

    def init_model(self, args):
        self.model = ConversationGraph(**vars(args)).to(self.device)

    def forward(self, inputs, target):
        output, a_vec, q_vec, mean_a_vec, mean_q_vec, mean_qa_vec, tilde_pre_vec = self.model(*inputs)
        target = target.view(-1).to(torch.float32)
        output = output.view(target.size(0), -1).to(torch.float32)
        if output.size(1) == 1:
            output = output.view(target.size(0))

        risk_loss = self.criterion_risk(output, target)

        backward_loss = risk_loss
        select_loss = torch.mean(risk_loss).view(1, 1)  # to select the best model

        return backward_loss, select_loss


class ConversationGraphTrainer(object):
    def __init__(self,
                 args,
                 config,
                 grad_clip=None,
                 patience_epochs=10
                 ):
        logging.info(f"initialize {self.__class__.__name__}")
        self.args = deepcopy(args)
        self.config = deepcopy(config)

        # tensorboard
        tb_path = os.path.join(self.args.result, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)
        os.makedirs(tb_path)
        self.tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)

        # model
        self.model_with_loss = ModelWrappedWithMSELoss(device=self.config.device)
        self.model_with_loss.init_model(self.config.model)
        self.model = self.model_with_loss.model

        logging.info(self.model)

        num_parameters = sum([l.nelement() for l in self.model.parameters()])
        logging.info("number of parameters: %d", num_parameters)

        # optim
        optim_params = vars(self.config.optim)
        if optim_params['optimizer'] == 'Adagrad':
            del optim_params['optimizer']
            optimizer = optim.Adagrad(self.model.parameters(), **optim_params)
        else:
            raise AssertionError('According to the original design, you should use "Adagrad" as the optimizer')
        self.optimizer = optimizer

    @staticmethod
    def __preprocess_data():
        """
        We omit this part, researches can change this part to their special domains
        :return:
        """
        pass

    def train(self):
        train_step = 0
        test_step = 0
        best_score = float('inf')

        # load data
        train_dataset = ConversationGraphDataset(
            ['data_2015.pkl',
             'data_2016.pkl'],
            label_name=self.config.data.label)
        eval_dataset = ConversationGraphDataset(
            ['data_2017.pkl'],
            label_name=self.config.data.label)

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=self.config.train.batch_size,
                                      shuffle=True,
                                      pin_memory=True,
                                      num_workers=0,
                                      drop_last=True)

        eval_dataloader = DataLoader(eval_dataset,
                                     batch_size=self.config.train.batch_size,
                                     shuffle=True,
                                     pin_memory=True,
                                     num_workers=0,
                                     drop_last=True)

        for epoch in range(self.config.train.n_epochs):
            logging.info(f"Epoch {epoch}")
            '''train'''
            self.model.train()
            total_train_loss = []
            for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                train_step += 1

                self.optimizer.zero_grad()

                model_input = [data['pre'].to(self.config.device),
                               # data['pre_num'],
                               data['qa'].to(self.config.device),
                               # data['qa_num'],
                               data['rule_mask'].to(self.config.device),
                               data['affinity_matrix'].to(self.config.device),
                               ]
                label = data['label'].to(self.config.device)

                backward_loss, train_select_loss = self.model_with_loss(model_input, label)
                backward_loss.backward()
                self.optimizer.step()
                total_train_loss.append(backward_loss.item())

                self.tb_logger.add_scalar('train_select_loss', train_select_loss, global_step=train_step)

            '''eval'''
            self.model.eval()
            with torch.no_grad():
                total_eval_loss = []
                cur_eval_score = []
                for i, data in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
                    test_step += 1

                    model_input = [data['pre'].to(self.config.device),
                                   # data['pre_num'],
                                   data['qa'].to(self.config.device),
                                   # data['qa_num'],
                                   data['rule_mask'].to(self.config.device),
                                   data['affinity_matrix'].to(self.config.device),
                                   ]
                    label = data['label'].to(self.config.device)

                    backward_loss, eval_select_loss = self.model_with_loss(model_input, label)
                    total_eval_loss.append(eval_select_loss.item())
                    cur_eval_score.append(eval_select_loss.item())
                    self.tb_logger.add_scalar('eval_select_loss', eval_select_loss, global_step=test_step)

            cur_score = np.mean(cur_eval_score)

            if cur_score < best_score:
                logging.info(f"best score is: {best_score}, current score is: {cur_score}, save best_checkpoint.pth")
                best_score = cur_score
                states = [
                    self.model.state_dict(),
                    self.optimizer.state_dict(),
                ]
                torch.save(states, os.path.join(self.args.checkpoint, 'best_checkpoint.pth'))

        # save checkpoint
        states = [
            self.model.state_dict(),
            self.optimizer.state_dict(),
        ]
        torch.save(states, os.path.join(self.args.checkpoint, 'checkpoint.pth'))

    def test(self, load_pre_train=True):
        if load_pre_train:
            # load pretrained_model
            pretrained_data = torch.load(os.path.join(self.args.checkpoint, 'best_checkpoint.pth'))
            self.model.load_state_dict(pretrained_data[0])

        test_dataset = ConversationGraphDataset(
            ['data_2018.pkl'],
            label_name=self.config.data.label)

        test_dataloader = DataLoader(test_dataset,
                                     batch_size=self.config.train.batch_size,
                                     shuffle=True,
                                     pin_memory=True,
                                     num_workers=0,
                                     drop_last=True)

        logging.info(f"Testing. Total {test_dataset.__len__()} data.")
        self.model.eval()
        output_list = []
        label_list = []

        with torch.no_grad():
            for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

                model_input = [data['pre'].to(self.config.device),
                               # data['pre_num'],
                               data['qa'].to(self.config.device),
                               # data['qa_num'],
                               data['rule_mask'].to(self.config.device),
                               data['affinity_matrix'].to(self.config.device),
                               ]
                label = data['label'].to(self.config.device)
                output = self.model(*model_input)
                for out in output.cpu().numpy():
                    output_list.append(out)
                for label_in in label.cpu().numpy():
                    label_list.append(label_in)

        mse = mean_squared_error(output.cpu().numpy(), label.cpu().numpy())
        mae = mean_absolute_error(output.cpu().numpy(), label.cpu().numpy())
        tau, _ = stats.kendalltau(label_list, output_list)
        rou, _ = stats.spearmanr(label_list, output_list)

        logging.info(f'label is {self.config.data.label}, '
                     f'mse is {mse}, '
                     f'mae is {mae}, '
                     f'rou is {rou}, '
                     f'tau is {tau}')

