from collections import defaultdict
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F 
import random

w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]

def read_data(filename):
  with open(filename) as f:
    for line in f:
      tag, words = line.lower().strip().split(" ||| ")
      yield ([w2i[x] for x in words.split(" ")], t2i[tag])

train = list(read_data("./data/sentence_classification/train.txt"))
dev = list(read_data("./data/sentence_classification/dev.txt"))

print(t2i)

EMB_SIZE = 64
WINDOW_SIZE = 3
FILTER_SIZE = 64
VOCAB_SIZE = len(w2i)

# class CNN(nn.Module):
#   def __init__(self, class_size, vocab_size, emb_size, window_size, filter_size):
#     super(CNN, self).__init__()
#     self.W_emb = nn.Embedding(vocab_size, emb_size)
#     self.conv = nn.Conv1d(1, filter_size, emb_size * window_size, stride = emb_size)
#     self.linear = nn.Linear()

