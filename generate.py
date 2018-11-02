#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-9-21 上午10:18
import torch
import pickle
import os
import json
from tqdm import tqdm
from model.model import RNNModel
from utils.utils import get_testloader

saves_dir = "./data/processed"  # directory to save processdata
model_save_dir = 'checkpoints/'  # directory to save model
gen_dir = 'gen/'  # directory to save generation definitions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

temperature = 1
CHAR_EMB_DIM = 20
CHAR_HID_DIM = 20


def generate(model, dataloader, idx2word, strategy='greedy', max_len=50):
    model.training = False
    for inp in tqdm(dataloader, desc='Generate Definitions', leave=False):
        word_list = []
        data = {
            'word': torch.tensor(inp['word'], dtype=torch.long).to(device),
            'seq': torch.tensor(torch.t(inp['seq']), dtype=torch.long).to(device),
            'chars': torch.tensor(inp['chars'], dtype=torch.long).to(device),
            'hnym': torch.tensor(inp['hnym'], dtype=torch.long).to(device),
            'hnym_weights': torch.tensor(inp['hnym_weights'], dtype=torch.float).to(device)
        }
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)
        def_word = [idx2word[torch.tensor(inp['word'], dtype=torch.long).to(device)], "\t"]
        word_list.extend(def_word)
        hidden = None
        for i in range(max_len):
            output, hidden = model(data, hidden)
            word_weights = output.squeeze().div(temperature).exp().cpu()
            if strategy == 'greedy':
                word_idx = torch.argmax(word_weights)
            elif strategy == 'multinomial':
                # 基于词的权重，对其再进行一次抽样，增添其多样性，如果不使用此法，会导致常用字的无限循环
                word_idx = torch.multinomial(word_weights, 1)[0]
            if word_idx == 3:
                break
            else:
                data['seq'].fill_(word_idx)
                if word_idx !=2:
                    word = idx2word[word_idx]
                    word_list.append(word)
        with open(gen_dir + "s+g.txt", "a") as f:
            for item in word_list:
                f.write(item + " ")
            f.write("\n")
            f.close()
    print("Finished!")
    return 1


if __name__ == "__main__":
    with open(os.path.join(saves_dir, 'word2id.pkl'), 'rb') as f:
        vocab = pickle.load(f)
        f.close()
    with open(os.path.join(saves_dir, 'id2word.pkl'), 'rb') as f:
        id2word = pickle.load(f)
        f.close()
    with open(os.path.join(saves_dir, 'word_embedding.pkl'), 'rb') as f:
        word_embedding = pickle.load(f)
        f.close()
    char2idx = json.loads(open('./data/processed/char2idx.js').read())
    char_data = {
        'char_vocab_size': len(char2idx) + 1,
        'char_emb_dim': CHAR_EMB_DIM,
        'char_hid_dim': CHAR_HID_DIM,
        'char_len': 21
    }
    test_loader = get_testloader('./data/processed/test.pkl')
    #   Model 需要和训练时候参数一致
    print('=========model architecture==========')
    model = RNNModel(
        'GRU', len(vocab), 300, 300, 2, False, 0.0, device, word_embedding,
        use_ch=False, use_he=False, use_i=False, use_h=False, use_g=True, **char_data).to(device)
    model_path = 's+g.pkl'
    model.load_state_dict(torch.load(model_save_dir + model_path), str(device))
    print(model)
    print('=============== end =================')
    generate(model, test_loader, id2word, max_len=25)
