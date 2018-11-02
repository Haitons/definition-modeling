#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-9-25 下午4:23
import codecs
from nltk.translate.bleu_score import corpus_bleu


def read_definition_file(def_file):
    defs = {}
    with codecs.open(def_file, 'r', 'utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            word = parts[0].strip()
            definition = parts[-1]
            if word not in defs:
                defs[word] = []
            defs[word].append(definition)
    return defs


def make_corpus(refs, hyps):
    refs_corpus = []
    hyps_corpus = []
    for word, definition in hyps.items():
        for d in definition:
            hyps_corpus.append(d.strip().split(' '))
            refs_corpus.append([h.split(' ') for h in refs[word]])
    return refs_corpus, hyps_corpus


def compute_bleu(ref_file, hyp_file):
    print("Reading files.")
    refs = read_definition_file(ref_file)
    hyps = read_definition_file(hyp_file)
    refs_corpus, hyps_corpus = make_corpus(refs, hyps)
    print('Computing BLEU...')
    bleu_corpus = corpus_bleu(refs_corpus, hyps_corpus,
                              (0.25, 0.25, 0.25, 0.25))
    print("Corpus Level BLEU: {}".format(bleu_corpus))
    bleu_1 = corpus_bleu(refs_corpus, hyps_corpus, (1, 0, 0, 0))
    print("1-gram BLEU: {}".format(bleu_1))
    bleu_2 = corpus_bleu(refs_corpus, hyps_corpus, (0, 1, 0, 0))
    print("2-gram BLEU: {}".format(bleu_2))
    bleu_3 = corpus_bleu(refs_corpus, hyps_corpus, (0, 0, 1, 0))
    print("3-gram BLEU: {}".format(bleu_3))
    bleu_4 = corpus_bleu(refs_corpus, hyps_corpus, (0, 0, 0, 1))
    print("4-gram BLEU: {}".format(bleu_4))


if __name__ == '__main__':
    ref_file = './data/commondefs/test.txt'
    hyp_file = './gen/generated.txt'
    compute_bleu(ref_file, hyp_file)

