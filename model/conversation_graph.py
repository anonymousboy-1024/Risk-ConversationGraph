import pickle

import numpy as np
import torch
from torch import nn
from torch.nn import init
import logging
from utils import *


def load_topic_rep(path):
    data = pickle.load(open(path, 'rb'))
    topic_rep = []
    for i in range(len(data.keys())):
        topic_rep.append(data[i])
    return np.array(topic_rep)


class ConversationGraph(nn.Module):
    def __init__(self,
                 hidden_size,
                 embedding_size=None,
                 hidden_layers=None,
                 dropout=None,
                 in_bn=True,
                 hid_bn=True,
                 out_bn=True,
                 make_ff=True,
                 ):
        super().__init__()
        logging.info(f"initialize {self.__class__.__name__}")

        # load pretrained topic representation
        lda_pre_rep = load_topic_rep("assets/lda_topic_rep.pkl")  # [28, d]
        self.topic_emb_dict = torch.tensor(lda_pre_rep).to(torch.float).cuda()
        self.topic_emb_dict.requires_grad = False

        self.embedding_size = embedding_size

        if make_ff:
            self.ff = self._make_ff(dropout,
                                    self.embedding_size,
                                    hidden_size,
                                    hidden_layers,
                                    in_bn=in_bn,
                                    hid_bn=hid_bn,
                                    out_bn=out_bn)

        self.sa = SimpleConcatAttention(emb_dim=self.embedding_size)  # can also use ScaledDotProductAttention

        self.ggnn = SimpleGatedGNN(input_dim=self.embedding_size, output_dim=self.embedding_size)



    def _make_ff(self, dropout, in_size, hidden_size, hidden_layers, in_bn=True, hid_bn=True, out_bn=True, out=True):
        def get_block(in_size, hidden_size, bn, act=True, drop=True):
            result = nn.Sequential(
                nn.BatchNorm1d(in_size) if bn else None,
                nn.Dropout(p=dropout) if drop else None,
                nn.Linear(in_size, hidden_size),
                nn.ReLU() if act else None,
            )
            return result

        ff_seq = list()
        ff_seq.extend(get_block(in_size, hidden_size[0], bn=in_bn))
        for i in range(1, hidden_layers):
            ff_seq.extend(get_block(hidden_size[i - 1], hidden_size[i], bn=hid_bn))
        if out:
            ff_seq.extend(get_block(hidden_size[-1], 1, bn=out_bn, act=False, drop=False))

        return Sequential(
            *ff_seq
        )

    def forward(self,
                pre_vec,
                qa_vec,
                rule_mask,
                affinity_matrix
                ):
        """
        :param pre_vec: [B, N^p, d]
        :param qa_vec:  [B, N^{qa}, d]
        :param rule_mask:  [B, N, N]
        :param affinity_matrix: [B, N, N]
        :return:
        """

        # generate \tilde{A}
        tilde_A = rule_mask * affinity_matrix

        # decompose the presentation section
        topic_pre_vec = self.sa(self.topic_emb_dict, pre_vec)  # [B, K, d]

        # update via conversation graph
        gnn_in = torch.cat((topic_pre_vec, qa_vec), dim=1)  # [B, N, d]
        v_prime = self.ggnn(tilde_A, gnn_in)  # [B, N, d]

        # graph readout
        v_prime_prime = torch.mean(v_prime, dim=1)  # [B, d]

        # risk prediciton
        y_hat = self.ff(v_prime_prime)

        return y_hat
