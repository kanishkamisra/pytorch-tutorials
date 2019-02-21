from gensim.models import KeyedVectors
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
import model

np.random.seed(1234)

tokenizer = RegexpTokenizer(r'\w+')
# Loading Data
w2i = defaultdict(lambda: len(w2i))
l2i = defaultdict(lambda: len(l2i))
UNK = w2i["<unk>"]

def read_snli(filename, indices = [0, 5, 6]):
  print("reading file: {}".format(filename))
  with open(filename, 'r') as f:
    next(f)
    for line in f:
      entry = line.split("\t")
      label, sentence1, sentence2 = [entry[i] for i in indices]
      if label == '-':
        continue
      else:
        sentence1, sentence2 = [tokenizer.tokenize(s.lower()) for s in [sentence1, sentence2]]
        # example entry in the data: ([1, 2, 3], [4, 2, 1], 1)
        yield ([w2i[x] for x in sentence1], [w2i[x] for x in sentence2], l2i[label])

files = ["train", "test", "dev"]

train, test, dev = [list(read_snli("../../snli_1.0/snli_1.0_{}.txt".format(f))) for f in files]

# print(l2i)

# load glove vector space and store word vectors
glove = KeyedVectors.load_word2vec_format("../../pretrained_vectors/glove_300d_word2vec.txt")
pretrained_glove = np.random.uniform(-0.25, 0.25, (len(w2i), 300))
print("GloVe embeddings vocab size: {}".format(len(pretrained_glove)))
pretrained_glove[0] = 0

for key in glove.vocab.keys():
  if key in w2i:
    pretrained_glove[w2i[key]] = glove[key]
  else:
    continue

decomp_model = model.Decomp(EMB_SIZE = 300, HIDDEN_SIZE = 100, OUTPUT_SIZE = 3, VOCAB = len(w2i))
decomp_model.embedding.weight.data.copy_(torch.from_numpy(pretrained_glove))
decomp_model.embedding.weight.requires_grad = False
