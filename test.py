#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-9-21 下午2:40
import os
import pickle
import torch
import numpy as np
import json
from torch import nn
from tqdm import tqdm
from model.model import RNNModel
from utils.dataset import DefSeqDataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

saves_dir = "./data/processed"  # directory to save processdata
model_save_dir = 'checkpoints/'  # directory to save model

CHAR_EMB_DIM = 20
CHAR_HID_DIM = 20

def get_dataloader(filepath):
    with open(filepath, 'rb') as f:
        valid = pickle.load(f)
    valid_set = DefSeqDataset(valid, mode='valid')
    char_max_len = len(valid_set[0]['chars'])
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=30,
        shuffle=True,
        num_workers=2
    )
    return valid_loader,char_max_len


def test(model, dataloader, device):
    model.training = False
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    with torch.no_grad():
        total_loss = []
        for inp in tqdm(dataloader, desc='Test model in the test set', leave=False):
            data = {
                'word': torch.tensor(inp['word'], dtype=torch.long).to(device),
                'seq': torch.tensor(torch.t(inp['seq']), dtype=torch.long).to(device),
                'chars': torch.tensor(inp['chars'], dtype=torch.long).to(device),
                'hnym': torch.tensor(inp['hnym'], dtype=torch.long).to(device),
                'hnym_weights': torch.tensor(inp['hnym_weights'], dtype=torch.float).to(device)
            }
            targets = torch.tensor(torch.t(inp['target']), dtype=torch.long).to(device)
            output, hidden = model(data, None)
            loss = loss_fn(output, targets.view(-1))
            total_loss.append(loss.item())
    return np.mean(total_loss), np.exp(np.mean(total_loss))

if __name__ == '__main__':
    with open(os.path.join(saves_dir, 'word2id.pkl'), 'rb') as f:
        vocab = pickle.load(f)
        f.close()
    with open(os.path.join(saves_dir, 'id2word.pkl'), 'rb') as f:
        id2word = pickle.load(f)
        f.close()
    with open(os.path.join(saves_dir, 'word_embedding.pkl'), 'rb') as f:
        word_embedding = pickle.load(f)
        f.close()
    test_loader,char_max_len = get_dataloader('./data/processed/test_full.pkl')
    char2idx = json.loads(open('./data/processed/char2idx.js').read())
    char_data = {
        'char_vocab_size': len(char2idx) + 1,
        'char_emb_dim': CHAR_EMB_DIM,
        'char_hid_dim': CHAR_HID_DIM,
        'char_len': char_max_len
    }


    #   Model 需要和训练时候参数一致
    print('=========model architecture==========')
    model = RNNModel(
        'GRU', len(vocab), 300, 300, 2, False, 0.0, device, word_embedding,
        use_ch=False,use_he=False,use_i=False,use_h=False,use_g=False,**char_data).to(device)
    model.load_state_dict(torch.load(model_save_dir + 'seed.pkl'),str(device))
    print(model)
    print('=============== end =================')
    loss, ppl = test(model, test_loader, device)
    print("The test set Loss:{0:>6.6},Ppl:{1:>6.6}".format(loss, ppl))
