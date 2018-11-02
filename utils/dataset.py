#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-9-4 下午3:37
from torch.utils.data import Dataset


class DefSeqDataset(Dataset):
    def __init__(self, data_file, mode='train'):
        self.mode = mode
        if self.mode not in ['train', 'valid', 'test']:
            raise Exception("Argument mode must be train,valid or test.")
        self.word = data_file['word']
        self.seq = data_file['seq']
        self.chars = data_file['chars']
        self.hynms = data_file['hynms']
        self.hynm_weight = data_file['hynm_weight']
        self.length = len(self.word)
        if not self.mode == 'test':
            self.target = data_file['target']

    def __getitem__(self, item):
        sample = {
            'word': self.word[item],
            'seq': self.seq[item],
            'chars': self.chars[item],
            'hnym': self.hynms[item],
            'hnym_weights': self.hynm_weight[item],
        }
        if not self.mode == 'test':
            sample['target'] = self.target[item]
        return sample

    def __len__(self):
        return self.length
