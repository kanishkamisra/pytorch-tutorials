from gensim.models import KeyedVectors
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer

indices = [0, 5, 6]
train = []
tokenizer = RegexpTokenizer(r'\w+')

for line in open("../snli_1.0/snli_1.0_test.txt"):
    entry = line.split("\t")
    label, sentence1, sentence2 = [entry[i] for i in indices]
    sentence1 = tokenizer.tokenize(sentence1.lower())
    sentence2 = tokenizer.tokenize(sentence2.lower())
    train.append((sentence1, sentence2, label))

