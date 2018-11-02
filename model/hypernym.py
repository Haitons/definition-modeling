#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-10-8 下午7:54
import torch
from torch import nn


class Hypernym(nn.Module):
    def __init__(self, embedding_dim, embeddings, device):
        super(Hypernym, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = embeddings
        self.device = device

    def forward(self, inputs):
        batch_hynm = inputs[0]
        batch_hynm_weights = inputs[1]
        batch_sum = []
        for hynm, weights in zip(batch_hynm, batch_hynm_weights):
            weighted_sum = torch.zeros(self.embedding_dim).to(self.device)
            for h, w in zip(hynm, weights):
                word_emb = self.embeddings(h)
                weighted_sum += w * word_emb
            batch_sum.append(torch.unsqueeze(weighted_sum, 0))
        return torch.cat(batch_sum, dim=0)
