#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-9-21 下午2:29
import pickle
import torch
import time
from utils.dataset import DefSeqDataset
from torch.utils.data import DataLoader
from datetime import timedelta


def get_testloader(filepath):
    with open(filepath, 'rb') as f:
        test = pickle.load(f)
    test_set = DefSeqDataset(test, mode='test')
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=True,
        num_workers=2
    )
    return test_loader

def repackage_hidden(h):
    """用新的变量重新包装隐藏层，将它们从历史中分离。"""
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))