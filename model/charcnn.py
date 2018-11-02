#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-8-28 下午2:32
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNN(nn.Module):
    def __init__(self, alphabet_size, pretrain_char_embedding, embedding_dim, hidden_dim, dropout):
        super(CharCNN, self).__init__()
        print("Build char sequence feature extractor: CNN ...")
        self.hidden_dim = hidden_dim
        self.char_drop = nn.Dropout(dropout)
        self.char_embeddings = nn.Embedding(alphabet_size, embedding_dim)
        if pretrain_char_embedding is not None:
            self.char_embeddings.weight.detach().copy_(
                torch.from_numpy(pretrain_char_embedding)
            )
        else:
            self.char_embeddings.weight.detach().copy_(
                torch.from_numpy(
                    self.random_embedding(alphabet_size, embedding_dim)
                )
            )
        self.char_cnn = nn.Conv1d(
            embedding_dim, self.hidden_dim, kernel_size=3, padding=1
        )
        nn.init.xavier_normal_(self.char_cnn.weight)
        nn.init.constant_(self.char_cnn.bias, 0.3)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for i in range(vocab_size):
            pretrain_emb[i, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def get_last_hiddens(self, input):
        """
                    input:
                        input: Variable(batch_size, word_length)
                        seq_lengths: numpy array (batch_size,  1)
                    output:
                        Variable(batch_size, char_hidden_dim)
                    Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
                """
        batch_size = input.size(0)
        char_embeds = self.char_drop(
            self.char_embeddings(input)
        )
        char_embeds = char_embeds.transpose(2, 1)
        char_cnn_out = self.char_cnn(char_embeds)
        char_cnn_out = F.max_pool1d(char_cnn_out, char_cnn_out.size(2)).reshape(batch_size, -1)
        return char_cnn_out

    def get_all_hiddens(self, x):
        """
            input:
                input: Variable(batch_size,  word_length)
                seq_lengths: numpy array (batch_size,  1)
            output:
                Variable(batch_size, word_length, char_hidden_dim)
            Note it only accepts ordered (length) variable, length size is recorded in seq_lengths
        """
        batch_size = x.size(0)
        char_embeds = self.char_drop(self.char_embeddings(x))
        char_embeds = char_embeds.transpose(2, 1)
        char_cnn_out = self.char_cnn(char_embeds).transpose(2, 1)
        char_cnn_out = char_cnn_out.reshape(batch_size, -1)
        return char_cnn_out

    def forward(self, x):
        return self.get_all_hiddens(x)


if __name__ == '__main__':
    device = torch.device('cuda')
    charcnn = CharCNN(500, None, 5, 4, 0).to(device)
    print(charcnn)
    inp = torch.randint(0, 499, (5, 10), dtype=torch.long).to(device)
    out = charcnn(inp)
    print(out,"aaaa")
    print(out.shape)
