#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-8-15 下午3:19
import os
import pickle
import codecs
import time
import json
import warnings
import numpy as np
from tqdm import tqdm
from datetime import timedelta
from gensim.models.keyedvectors import KeyedVectors
from collections import Counter, defaultdict

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

main_dir = "../data/"  # main directory
data_dir = "../data/commondefs"  # data directory
save_dir = "../data/processed"  # directory to save processdata

pad = '<pad>'  # pad symbol
uns = '<unk>'  # unknown symbol
starts = '<def>'  # start sentence symbol
ends = '</s>'  # end sentence symbol

embedding_dim = 300
embedding_path = os.path.join(main_dir, 'word2vec/GoogleNews-vectors-negative300.bin')


# Get time
def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


# Read data
def read_data(file_path):
    content = []
    token = []
    with codecs.open(file_path, 'r', 'utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            token.append(line[0])
            content.append((line[0], line[-1]))
            definition = line[-1].strip().split(' ')
            for d in (definition):
                token.append(d)
    return content, token


# Read hypernyms
def read_hypernyms(file_path):
    hyp_token = []
    hnym_data = defaultdict(list)
    with open(file_path, 'r') as fr:
        for line in fr:
            line = line.strip().split('\t')
            word = line[0]
            hyp_token.append(word)
            line = line[1:]
            assert len(line) % 2 == 0
            for i in range(int(len(line) / 2)):
                hnym = line[2 * i]
                hyp_token.append(hnym)
                weight = line[2 * i + 1]
                hnym_data[word].append((hnym, weight))
    return hnym_data, hyp_token


# Get hypernyms
def get_hnym(hnym_data, word2idx):
    word2hnym = defaultdict(list)
    hnym_weights = defaultdict(list)
    for key, value in hnym_data.items():
        weight_sum = sum([float(w) for h, w in value])
        for hnym, weight in value:
            word2hnym[key].append(word2idx[hnym])
            hnym_weights[key].append(float(weight) / weight_sum)
    return word2hnym, hnym_weights


# Build vocabulary
def build_vocab(train_data, valid_data, test_data, hnym_data):
    start_time = time.time()
    print("Start build the vocab in {}".format(time.asctime(time.localtime(start_time))))
    word_counts = Counter()
    word_counts.update(train_data)
    word_counts.update(valid_data)
    word_counts.update(test_data)
    word_counts.update(hnym_data)

    # index2word
    vocabulary_inv = []
    vocabulary_inv.extend([pad, uns, starts, ends])
    for x in tqdm(word_counts.most_common()):
        if x[0] not in vocabulary_inv:
            vocabulary_inv.append(x[0])

    # word2index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    time_dif = get_time_dif(start_time)
    print("Finished!Build vocab time usage:", time_dif)
    return [vocabulary, vocabulary_inv]


# Build char2id
def build_char2id(vocab):
    char2id = {'<c>': 1, '</c>': 2}
    for key in vocab.keys():
        for c in key:
            if c not in char2id:
                char2id[c] = len(char2id) + 1
    return char2id


# Load embeddings
def load_embeddings(embedding_path, vocab, embedding_dim):
    start_time = time.time()
    print('Load word2vecs from file {} in {}'.format(embedding_path, time.asctime(time.localtime(start_time))))
    word2vec = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
    init_embedding = np.random.uniform(-1.0, 1.0, (len(vocab), embedding_dim))
    print()
    for word in tqdm(vocab):
        if word in word2vec:
            init_embedding[vocab[word]] = word2vec[word]
    init_embedding[vocab['<pad>']] = np.zeros([embedding_dim])
    time_dif = get_time_dif(start_time)
    print("Finished!Load word embedding time usage:", time_dif)
    return init_embedding


# Build data
def bulid_data(train_data, valid_data, test_data, vocab):
    start_time = time.time()
    print('Build data in {}'.format(time.asctime(time.localtime(start_time))))
    train_word = np.zeros(len(train_data))
    valid_word = np.zeros(len(valid_data))
    test_word = np.zeros(len(test_data))
    print("First:build word.")
    for data, norm_word in zip(
            [train_data, valid_data, test_data], [train_word, valid_word, test_word]):
        assert len(data) == len(norm_word)
        for i, (word, _) in tqdm(enumerate(data)):
            norm_word[i] = vocab[word]
    print("Second:build sequence.")
    max_len = 0
    for data in [train_data, valid_data]:
        for _, seq in data:
            list = seq.split(" ")
            max_len = max(max_len, len(list) + 3)
    train_seq = np.zeros((len(train_data), max_len))
    valid_seq = np.zeros((len(valid_data), max_len))
    for data, norm_seq in zip(
            [train_data, valid_data], [train_seq, valid_seq]):
        assert len(data) == len(norm_seq)
        for i, (word, seq) in tqdm(enumerate(data)):
            seq = seq.split(" ")
            seq.insert(0, '<def>')
            seq.insert(0, word)
            seq.append('</s>')
            for j, k in enumerate(seq):
                norm_seq[i][j] = vocab[k]
    test_seq = np.zeros((len(test_data), 1))
    for i, (word, _) in tqdm(enumerate(test_data)):
        test_seq[i] = np.array([vocab[word]])
    time_dif = get_time_dif(start_time)
    print("Finished!Build data time usage:", time_dif)
    return [train_word, valid_word, test_word], [train_seq, valid_seq, test_seq]


# Prepare chars
def prep_chars(train, test, valid, char2idx):
    start_time = time.time()
    print('Preparing Characters in {}'.format(time.asctime(time.localtime(start_time))))
    max_len = 0
    for data in [train, test, valid]:
        for word, _ in data:
            word_len = len(word) + 2
            max_len = max(word_len, max_len)
    train_chars = np.zeros((len(train), max_len))
    test_chars = np.zeros((len(test), max_len))
    valid_chars = np.zeros((len(valid), max_len))
    for data, norm_chars in zip([train, test, valid],
                                [train_chars, test_chars, valid_chars]):
        assert len(data) == len(norm_chars)
        for i, (word, _) in tqdm(enumerate(data), leave=False):
            chars = [c for c in word]
            chars.insert(0, '<c>')
            chars.append('</c>')
            for j, c in enumerate(chars):
                norm_chars[i][j] = char2idx[c]
    time_dif = get_time_dif(start_time)
    print("Finished!Prepare chars time usage:", time_dif)
    return train_chars, test_chars, valid_chars


# Prepare hnyms
def prep_hnym(train, test, valid, word2hnym, hnym_weights, top_k=5):
    start_time = time.time()
    print('Preparing Hypernyms in {}'.format(time.asctime(time.localtime(start_time))))
    train_hnym = np.zeros((len(train), top_k))
    test_hnym = np.zeros((len(test), top_k))
    valid_hnym = np.zeros((len(valid), top_k))
    train_hnym_weights = np.zeros_like(train_hnym)
    test_hnym_weights = np.zeros_like(test_hnym)
    valid_hnym_weights = np.zeros_like(valid_hnym)
    for data, norm_hnym, norm_hnym_weights in zip(
            [train, test, valid], [train_hnym, test_hnym, valid_hnym],
            [train_hnym_weights, test_hnym_weights, valid_hnym_weights]):
        assert len(data) == len(norm_hnym)
        for i, (word, _) in tqdm(enumerate(data), leave=False):
            for j, hnym in enumerate(word2hnym[word][:top_k]):
                norm_hnym[i][j] = hnym
            for k, weight in enumerate(hnym_weights[word][:top_k]):
                norm_hnym_weights[i][k] = weight
    time_dif = get_time_dif(start_time)
    print("Finished!Prepare Hypernyms time usage:", time_dif)
    return train_hnym, train_hnym_weights, test_hnym, test_hnym_weights, valid_hnym, valid_hnym_weights


# Build target
def build_target(train_data, valid_data, vocab):
    start_time = time.time()
    print('Build target in {}'.format(time.asctime(time.localtime(start_time))))
    max_len = 0
    for data in [train_data, valid_data]:
        for _, seq in data:
            list = seq.split(" ")
            max_len = max(max_len, len(list) + 3)
    train_target = np.zeros((len(train_data), max_len))
    valid_target = np.zeros((len(valid_data), max_len))
    for data, norm_target in zip(
            [train_data, valid_data], [train_target, valid_target]):
        assert len(data) == len(norm_target)
        for i, (_, seq) in tqdm(enumerate(data)):
            seq = seq.split(" ")
            seq.insert(0, '<def>')
            seq.append('</s>')
            for j, k in enumerate(seq):
                norm_target[i][j] = vocab[k]
    time_dif = get_time_dif(start_time)
    print("Finished!Build target time usage:", time_dif)
    return [train_target, valid_target]


if __name__ == "__main__":
    # First:collect tokens,build word2index and index2id
    train_file = os.path.join(data_dir, "train.txt")
    valid_file = os.path.join(data_dir, "valid.txt")
    test_file = os.path.join(data_dir, "test.txt")
    test_short_list_file = os.path.join(data_dir, "./shortlist/shortlist_test.txt")
    hypernym_file = os.path.join(data_dir, "./auxiliary/bag_of_hypernyms.txt")

    train_data, train_token = read_data(train_file)
    valid_data, valid_token = read_data(valid_file)
    test_data, test_token = read_data(test_file)
    test_short_list, _ = read_data(test_short_list_file)
    hypernym_data, hyp_token = read_hypernyms(hypernym_file)

    vocab, vocab_inv = build_vocab(train_token, valid_token, test_token, hyp_token)
    char2id = build_char2id(vocab)
    print("Bulid vocab finished,Vocab size:%d" % len(vocab))

    # Second:build word2vec
    init_embedding = load_embeddings(embedding_path, vocab, embedding_dim)

    # Third:build data
    word2hym, hym_weights = get_hnym(hypernym_data, vocab)
    word, sequence = bulid_data(train_data, valid_data, test_short_list, vocab)
    target = build_target(train_data, valid_data, vocab)
    train_chars, test_chars, valid_chars = prep_chars(train_data, test_short_list, valid_data, char2id)
    train_hnym, train_hnym_weights, test_hnym, test_hnym_weights, valid_hnym, valid_hnym_weights = prep_hnym(
        train_data, test_short_list, valid_data, word2hym, hym_weights, )

    print("Saving.....")
    # Save word2id,id2word and char2id
    with open(os.path.join(save_dir, 'word2id.pkl'), 'wb') as f:
        pickle.dump(vocab, f)
        f.close()
    with open(os.path.join(save_dir, 'id2word.pkl'), 'wb') as f:
        pickle.dump(vocab_inv, f)
        f.close()
    with open(os.path.join(save_dir, 'char2id.pkl'), 'wb') as f:
        pickle.dump(char2id, f)
        f.close()

    # Save word embedding
    with open(os.path.join(save_dir, 'word_embedding.pkl'), 'wb') as f:
        pickle.dump(init_embedding, f)
        f.close()

    # Save data
    train = {'word': word[0], 'seq': sequence[0], 'target': target[0], 'chars': train_chars, 'hynms': train_hnym,
             'hynm_weight': train_hnym_weights}
    valid = {'word': word[1], 'seq': sequence[1], 'target': target[1], 'chars': valid_chars, 'hynms': valid_hnym,
             'hynm_weight': valid_hnym_weights}
    test = {'word': word[2], 'seq': sequence[2], 'chars': test_chars, 'hynms': test_hnym,
            'hynm_weight': test_hnym_weights}
    with open(os.path.join(save_dir, 'train.pkl'), 'wb') as f:
        pickle.dump(train, f)
        f.close()
    with open(os.path.join(save_dir, 'valid.pkl'), 'wb') as f:
        pickle.dump(valid, f)
        f.close()
    with open(os.path.join(save_dir, 'test.pkl'), 'wb') as f:
        pickle.dump(test, f)
        f.close()

    # Save json
    with open(os.path.join(save_dir, 'word2idx.js'), 'w') as fr1:
        fr1.write(json.dumps(vocab))
    with open(os.path.join(save_dir, 'word2hnym.js'), 'w') as fr2:
        fr2.write(json.dumps(word2hym))
    with open(os.path.join(save_dir, 'char2idx.js'), 'w') as fr3:
        fr3.write(json.dumps(char2id))
    with open(os.path.join(save_dir, 'hnym_weights.js'), 'w') as fr5:
        fr5.write(json.dumps(hym_weights))

    # with open(os.path.join(save_dir, 'vocabulary.pkl'), 'rb') as f:
    #     vocab=pickle.load(f)
    #     f.close()
    # with open(os.path.join(save_dir, 'vocabulary_inv.pkl'), 'rb') as f:
    #     vocab_inv=pickle.load(f)
    #     f.close()
    # with open(os.path.join(save_dir, 'train.pkl'), 'rb') as f:
    #     train=pickle.load(f)
    #     f.close()
    # with open(os.path.join(save_dir, 'valid.pkl'), 'rb') as f:
    #     valid=pickle.load(f)
    #     f.close()
    # with open(os.path.join(save_dir, 'test.pkl'), 'rb') as f:
    #     test=pickle.load(f)
    #     f.close()

    print("Over")
