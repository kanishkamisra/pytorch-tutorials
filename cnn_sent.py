from collections import defaultdict
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F 
import random
import numpy as np
from sklearn.utils import shuffle

w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]

def read_data(filename):
  with open(filename) as f:
    for line in f:
      tag, words = line.lower().strip().split(" ||| ")
      yield ([w2i[x] for x in words.split(" ")], t2i[tag])

train = list(read_data("./data/sentence_classification/train.txt"))
print(len(w2i))
dev = list(read_data("./data/sentence_classification/dev.txt"))
print(len(w2i))
test = list(read_data("./data/sentence_classification/test.txt"))
print(len(w2i))
# print(t2i)

EMB_SIZE = 64
WINDOW_SIZE = 3
FILTER_SIZE = 64
VOCAB_SIZE = len(w2i)
MAX_SENT_LEN = max([len(k) for k, v in train + dev + test])

class CNN(nn.Module):
  def __init__(self, **kwargs):
    super(CNN, self).__init__()

    # self.MODEL = kwargs["MODEL"]  useful when I do Kim 2014 implementation
    self.CLASS_SIZE = kwargs["CLASS_SIZE"]
    self.BATCH_SIZE = kwargs["BATCH_SIZE"]
    self.EMB_DIM = kwargs["EMB_DIM"]
    self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
    self.MAX_LEN = kwargs["MAX_LEN"]
    self.WINDOWS = kwargs["WINDOWS"]
    self.FILTER_MAPS = kwargs["FILTER_MAPS"]
    self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
    # self.IN_CHANNEL = kwargs["IN_CHANNEL"] useful for reproducing Kim 2014

    self.W_emb = nn.Embedding(self.VOCAB_SIZE + 2, self.EMB_DIM, padding_idx = self.VOCAB_SIZE + 1)

    for i in range(len(self.WINDOWS)):
      conv = nn.Conv1d(1, self.FILTER_MAPS[i], self.EMB_DIM * self.WINDOWS[i])
      setattr(self, f'conv_{i}', conv)
    
    self.linear = nn.Linear(sum(self.FILTER_MAPS), self.CLASS_SIZE)
  
  def get_convolution(self, i):
    return(getattr(self, f'conv_{i}'))

  def forward(self, input):
    x = self.embedding(input).view(-1, 1, self.EMB_DIM * self.MAX_LEN)
    conv_results = [F.max_pool1d(F.relu(self.get_conv(i)(x)), self.MAX_LEN - self.WINDOWS[i] + 1).view(-1, self.FILTER_MAPS[i]) for i in range(len(self.WINDOWS))]

    x = torch.cat(conv_results, 1)
    x = F.dropout(x, p = self.DROPOUT_PROB, training = self.training)
    x = self.linear(x)

    # return(F.log_softmax(x, dim = 1))
    return(x)

