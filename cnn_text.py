from collections import defaultdict
import time
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):
  def __init__(self, nwords, emb_size, filter_size, window_size, class_size):
    super(CNN, self).__init__()

    # Embedding layer
    self.embedding = nn.Embedding(nwords, emb_size)

    # uniform initialization 
    nn.init.uniform_(self.embedding.weight, -0.25, 0.25)

    # 1d convolutions
    self.conv_1d = nn.Conv1d(in_channels = emb_size, out_channels = filter_size, kernel_size = window_size, stride = 1, padding = 0, dilations = 1, groups = 1, bias = True)

    # relu unit
    self.relu = nn.ReLU()

    # projection layer
    self.projection_layer = nn.Linear(in_features = filter_size, out_features = class_size, bias = True)

    # xavier initialization of the projection layer
    nn.init.xavier_uniform_(self.projection_layer.weight)

  def forward(self, words):
    emb = self.embedding(words) # nwords x emb_size
    emb = emb.unsqueeze(0).permute(0, 2, 1) # 1 x emb_size x vocab
    h = self.conv_1d(emb) # perform convolution over the stretched embeddings

    # max pooling
    h = h.max(dim = 2)[0]
    h = self.relu(h)
    out = self.projection_layer(h)
    return out

w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]

def read_data(filename):
  print("Reading {}..".format(filename))
  with open(filename) as f:
    for line in f:
      tag, words = line.lower().strip().split(" ||| ")
      yield ([w2i[x] for x in words.split(" ")], t2i[tag])

files = ["train", "dev", "test"]

train, dev, test = [list(read_data("./data/sentence_classification/{}.txt".format(f))) for f in files]

nwords = len(w2i)
CLASS_SIZE = len(t2i)
EMB_SIZE = 64
WIN_SIZE = 3
FILTER_SIZE = 64
# print("number of words = {}, class size = {}.".format(nwords, CLASS_SIZE))

# initialize the model
model = CNN(nwords, EMB_SIZE, FILTER_SIZE, WIN_SIZE, ntags)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mode.parameters())

type = torch.LongTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
  type = torch.cuda.LongTensor
  model.cuda()

for ITER in range(100):
  random.shuffle(train)
  train_loss = 0.0
  train_correct = 0.0
  start = time.time()
  for words, tag in train:
    if len(words) < WIN_SIZE:
      words += [0] * (WIN_SIZE - len(words))
    words_tensor = torch.tensor(words).type(type)
    tag_tensor = torch.tensor([tag]).type(type)
    scores = model(words_tensor)
    predict = scores[0].argmax().items()
    if predict == tag:
      train_correct += 1

    loss = criterion(scores, tag_tensor)
    train_loss += my_loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

