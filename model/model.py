#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-9-11 下午11:30
import torch
from torch import nn
from model.charcnn import CharCNN
from model.hypernym import Hypernym
import torch.nn.functional as F
import numpy as np


class RNNModel(nn.Module):
    def __init__(self, rnn_type, vocab_size, embedding_dim, hidden_dim, num_layers,
                 tie_weights, dropout, device, pretrain_emb=None,
                 use_ch=False, use_he=False, use_i=False, use_h=False, use_g=True, **kwargs):
        super(RNNModel, self).__init__()
        self.rnn_type = rnn_type
        self.n_layers = num_layers
        self.hi_dim = hidden_dim

        self.device = device
        self.use_i = use_i
        self.use_h = use_h
        self.use_g = use_g
        self.use_ch = use_ch
        self.use_he = use_he

        self.drop = nn.Dropout(dropout)

        char_hid_dim = 0
        char_len = 0
        he_dim = 0

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrain_emb is not None:
            self.embedding.weight.data.copy_(
                torch.from_numpy(pretrain_emb)
            )
        else:
            self.embedding.weight.data.copy_(
                torch.from_numpy(
                    self.random_embedding(vocab_size, embedding_dim)
                )
            )
        self.embedding.weight.requires_grad = False
        # ch
        if use_ch:
            char_vocab_size = kwargs['char_vocab_size']
            char_emb_dim = kwargs['char_emb_dim']
            char_hid_dim = kwargs['char_hid_dim']
            char_len = kwargs['char_len']
            self.ch = CharCNN(char_vocab_size, None, char_emb_dim, char_hid_dim, dropout).to(device)
        # he
        if use_he:
            print("Build Hypernym Embeddings...")
            he_dim = embedding_dim
            self.he = Hypernym(embedding_dim, self.embedding, device)
        concat_embedding_dim = embedding_dim + char_len * char_hid_dim + he_dim
        if self.use_i:
            embedding_dim = embedding_dim + concat_embedding_dim
        if self.use_h:
            self.h_linear = nn.Linear(concat_embedding_dim + hidden_dim, hidden_dim)
        if self.use_g:
            self.zt_linear = nn.Linear(concat_embedding_dim + hidden_dim, hidden_dim)
            self.rt_linear = nn.Linear(concat_embedding_dim + hidden_dim, concat_embedding_dim)
            self.ht_linear = nn.Linear(concat_embedding_dim + hidden_dim, hidden_dim)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(embedding_dim, hidden_dim, num_layers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                               options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, nonlinearity=nonlinearity, dropout=dropout)
        self.word2hidden = nn.Linear(concat_embedding_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, vocab_size)
        if tie_weights:
            if hidden_dim != embedding_dim:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.embedding.weight
        self.init_weights()

    def forward(self, inputs, init_hidden):
        word = inputs['word']
        seq = inputs['seq']
        chars = inputs['chars']
        hynm = inputs['hnym']
        hynm_weights = inputs['hnym_weights']
        batch_size = word.size(0)

        word_emb = self.embedding(word)
        seq_emb = self.embedding(seq)
        if self.use_ch:
            char_embeddings = self.ch(chars)
            word_emb = torch.cat(
                [word_emb, char_embeddings], dim=-1)
        if self.use_he:
            hynm_embeddings = self.he([hynm, hynm_weights])
            word_emb = torch.cat(
                [word_emb, hynm_embeddings], dim=-1)

        if init_hidden is not None:
            hidden = init_hidden
        else:
            hidden=None
        # else:
        #     hidden = self.word2hidden(word_emb).view(-1, batch_size, self.hi_dim)
            # hidden = self.drop(hidden)
            # hidden = hidden.expand(self.n_layers, batch_size, self.hi_dim).contiguous()
            # if self.rnn_type == 'LSTM':
            #     h_c = hidden
            #     h_h = torch.zeros_like(h_c)
            #     hidden = (h_h, h_c)
        outputs = []
        for time_step in range(seq.size(0)):
            inp_seq = seq_emb[time_step, :, :].view(1, batch_size, -1)
            if self.use_i:
                inp_seq = torch.cat([torch.unsqueeze(word_emb, 0), inp_seq], dim=-1)
                outs, hidden = self.rnn(inp_seq, hidden)
            else:
                outs, hidden = self.rnn(inp_seq, hidden)
                if self.use_h:
                    if self.rnn_type == 'LSTM':
                        inp_h = torch.cat(
                            [torch.unsqueeze(word_emb, 0).expand(self.n_layers, batch_size, -1).contiguous(),
                             hidden[0]], dim=-1)
                        # inp_c = torch.cat(
                        #     [torch.unsqueeze(word_emb, 0).expand(self.n_layers, batch_size, -1).contiguous(),
                        #      hidden[1]], dim=-1)
                        hidden = (F.tanh(self.h_linear(inp_h)), hidden[1])
                        # hidden = (F.tanh(self.h_linear(inp_h)), F.tanh(self.h_linear(inp_c)))
                    else:
                        inp_h = torch.cat(
                            [torch.unsqueeze(word_emb, 0).expand(self.n_layers, batch_size, -1).contiguous(), hidden],
                            dim=-1)
                        hidden = F.tanh(self.h_linear(inp_h))
                if self.use_g:
                    if self.rnn_type == 'LSTM':
                        inp_h = torch.cat(
                            [torch.unsqueeze(word_emb, 0).expand(self.n_layers, batch_size, -1).contiguous(),
                             hidden[0]], dim=-1)
                        z_t = F.sigmoid(self.zt_linear(inp_h))
                        r_t = F.sigmoid(self.rt_linear(inp_h))
                        mul = torch.mul(r_t, word_emb)
                        hidden_ = torch.cat([mul, hidden[0]], dim=-1)
                        hidden_ = F.tanh(self.ht_linear(hidden_))
                        hidden = (torch.mul((1 - z_t), hidden[0]) + torch.mul(z_t, hidden_), hidden[1])
                    else:
                        inp_h = torch.cat(
                            [torch.unsqueeze(word_emb, 0).expand(self.n_layers, batch_size, -1).contiguous(), hidden],
                            dim=-1)
                        z_t = F.sigmoid(self.zt_linear(inp_h))
                        r_t = F.sigmoid(self.rt_linear(inp_h))
                        mul = torch.mul(r_t, word_emb)
                        hidden_ = torch.cat([mul, hidden], dim=-1)
                        hidden_ = F.tanh(self.ht_linear(hidden_))
                        hidden = torch.mul((1 - z_t), hidden) + torch.mul(z_t, hidden_)
            outputs.append(outs)
        outputs = torch.cat(outputs, dim=0)
        outputs = outputs.view(outputs.size(0) * outputs.size(1), outputs.size(2))
        decoded = self.decoder(self.drop(outputs))
        return decoded, hidden

    def init_weights(self):
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        nn.init.constant_(self.word2hidden.bias, 0.0)
        nn.init.xavier_normal_(self.word2hidden.weight)
        nn.init.constant_(self.decoder.bias, 0.0)
        nn.init.xavier_normal_(self.decoder.weight)
        if self.use_h:
            nn.init.constant_(self.h_linear.bias, 0.0)
            nn.init.xavier_normal_(self.h_linear.weight)
        if self.use_g:
            nn.init.constant_(self.zt_linear.bias, 0.0)
            nn.init.xavier_normal_(self.zt_linear.weight)
            nn.init.constant_(self.rt_linear.bias, 0.0)
            nn.init.xavier_normal_(self.rt_linear.weight)
            nn.init.constant_(self.ht_linear.bias, 0.0)
            nn.init.xavier_normal_(self.ht_linear.weight)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for i in range(vocab_size):
            pretrain_emb[i, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.n_layers, bsz, self.hi_dim),
                    weight.new_zeros(self.n_layers, bsz, self.hi_dim))
        else:
            return weight.new_zeros(self.n_layers, bsz, self.hi_dim)
