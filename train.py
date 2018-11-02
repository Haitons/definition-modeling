#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-9-13 下午1:35
import os
import pickle
import numpy as np
import torch
import time
import argparse
import json
from torch import nn
from tqdm import tqdm
from model.model import RNNModel
from utils.dataset import DefSeqDataset
from torch.utils.data import DataLoader
from utils.utils import get_testloader, get_time_dif

parser = argparse.ArgumentParser(description='Pytorch Definition Sequence Model')
parser.add_argument('--model', type=str, default='GRU',
                    help='type of recurrent network(RNN,LSTM,GRU)')
parser.add_argument('--emdim', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--hidim', type=int, default=300,
                    help='numbers of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=int, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=int, default='5',
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=50,
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers(0 = no dropout)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--use_i', action='store_true',
                    help='use model I')
parser.add_argument('--use_h', action='store_true',
                    help='use model H')
parser.add_argument('--use_g', action='store_true',
                    help='use model G')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=22222,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--use_ch', action='store_true',
                    help='use character level CNN')
parser.add_argument('--use_he', action='store_true',
                    help='use hypernym embeddings')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
args = parser.parse_args()
args.tied = False

args.use_ch = False
args.use_he = False
args.use_i = False
args.use_h = False
args.use_g = True

CHAR_EMB_DIM = 20
CHAR_HID_DIM = 20

saves_dir = "./data/processed"  # directory to save processdata
model_save_dir = 'checkpoints/'  # directory to save model
gen_dir = 'gen/'  # directory to save generation definitions

# Set the random seed manually for reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING:You have a CUDA device,so you should probably run with --cuda')
    else:
        torch.cuda.manual_seed(args.seed)
device = torch.device('cuda' if args.cuda else 'cpu')


def get_trainloader(filepath):
    with open(filepath, 'rb') as f:
        train = pickle.load(f)
    train_set = DefSeqDataset(train, mode='train')
    char_max_len = len(train_set[0]['chars'])
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    return train_loader, char_max_len


def get_validloader(filepath):
    with open(filepath, 'rb') as f:
        valid = pickle.load(f)
    valid_set = DefSeqDataset(valid, mode='valid')
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    return valid_loader


def train():
    with open(os.path.join(saves_dir, 'word2id.pkl'), 'rb') as f:
        vocab = pickle.load(f)
        f.close()
    with open(os.path.join(saves_dir, 'id2word.pkl'), 'rb') as f:
        id2word = pickle.load(f)
        f.close()
    with open(os.path.join(saves_dir, 'word_embedding.pkl'), 'rb') as f:
        word_embedding = pickle.load(f)
        f.close()
    train_loader, char_max_len = get_trainloader('./data/processed/train.pkl')
    valid_loader = get_validloader('./data/processed/valid.pkl')
    # test_loader = get_testloader('./data/processed/play.pkl')
    char2idx = json.loads(open('./data/processed/char2idx.js').read())
    char_data = {
        'char_vocab_size': len(char2idx) + 1,
        'char_emb_dim': CHAR_EMB_DIM,
        'char_hid_dim': CHAR_HID_DIM,
        'char_len': char_max_len
    }
    print('=========model architecture==========')
    model = RNNModel(
        args.model, len(vocab), args.emdim, args.hidim, args.nlayers, args.tied, args.dropout, device, word_embedding,
        use_ch=args.use_ch, use_he=args.use_he, use_i=args.use_i, use_h=args.use_h, use_g=args.use_g, **char_data).to(
        device)
    print(model)
    print('=============== end =================')
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, args.lr,weight_decay=args.wdecay)
    print('Training and evaluating...')
    start_time = time.time()
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    best_ppl = 9999999
    last_improved = 0
    require_improvement = 5
    for epoch in range(args.epochs):
        model.training = True
        total_loss = 0.0
        loss_epoch = []
        for batch, inp in enumerate(tqdm(train_loader, desc='Epoch: %03d' % (epoch + 1), leave=False)):
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
            optimizer.zero_grad()
            total_loss += loss.item()
            loss_epoch.append(loss.item())
            loss.backward()
            # `clip_grad_norm`
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
        train_loss = np.mean(loss_epoch)
        train_ppl = np.exp(train_loss)
        valid_loss, valid_ppl = evaluate(model, valid_loader, device)
        # generate(model, test_loader, id2word, epoch + 1, 20)

        if valid_ppl < best_ppl:
            best_ppl = valid_ppl
            last_improved = epoch
            torch.save(model.state_dict(), model_save_dir +
                       'defseq_model_params_%s_min_ppl.pkl' % (epoch + 1)
                       )
            improved_str = '*'
        else:
            torch.save(model.state_dict(), model_save_dir +
                       'defseq_model_params_%s_ppl.pkl' % (epoch + 1)
                       )
            improved_str = ''
        time_dif = get_time_dif(start_time)
        msg = 'Epoch: {0:>6},Train Loss: {1:>6.6}, Train Ppl: {2:>6.6},' \
              + ' Val loss: {3:>6.6}, Val Ppl: {4:>6.6},Time:{5} {6}'
        print(msg.format(epoch + 1, train_loss, train_ppl, valid_loss, valid_ppl, time_dif, improved_str))
        if epoch - last_improved > require_improvement:
            print("No optimization for a long time, auto-stopping...")
            break
    return 1


def evaluate(model, dataloader, device='cpu'):
    model.training = False
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    with torch.no_grad():
        total_loss = []
        for inp in dataloader:
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


def generate(model, dataloader, idx2word, num, word_len=50):
    model.training = False
    for inp in tqdm(dataloader, desc='Generate definitions', leave=False):
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
        for i in range(word_len):
            output, hidden = model(data, hidden)
            word_weights = output.squeeze().div(args.temperature).exp().cpu()
            word_idx = torch.argmax(word_weights)
            if word_idx == 3:
                break
            else:
                data['seq'].fill_(word_idx)
                if word_idx != 2:
                    word = idx2word[word_idx]
                    word_list.append(word)
        with open(gen_dir + "gen_%s.txt" % (num), "a") as f:
            for item in word_list:
                f.write(item + " ")
            f.write("\n")
            f.close()
    return 1


if __name__ == "__main__":
    print('=============user config=============')
    for key, value in sorted(args.__dict__.items()):
        print('{key}:{value}'.format(key=key, value=value))
    print('=============== end =================')
    train()
