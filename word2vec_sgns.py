import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F 
import nltk
import random
import numpy as np 
from collections import Counter

flatten = lambda l: [item for sublist in l for item in sublist]
random.seed(1024)
torch.manual_seed(1)

def get_batch(batch_size, train_data):
  random.shuffle(train_data)
  sindex = 0
  eindex = batch_size
  while eindex < len(train_data):
    batch = train_data[sindex:eindex]
    temp = eindex
    eindex = eindex + batch_size
    sindex = temp
    yield batch

  if eindex >= len(train_data):
    batch = train_data[sindex:]
    yield batch

corpus = list(nltk.corpus.gutenberg.sents('melville-moby_dick.txt'))[:500]
corpus = [[word.lower() for word in sent] for sent in corpus]

word_count = Counter(flatten(corpus))

MIN_COUNT = 3
exclude = []

for w, c in word_count.items():
  if c < MIN_COUNT:
    exclude.append(w)

## Remove words having frequency less than 3.
vocab = list(set(flatten(corpus)) - set(exclude))
vocab.append('<UNK>')

word_to_ix = {'<UNK>': 0}
for word in vocab:
  if word_to_ix.get(word) is None:
    word_to_ix[word] = len(word_to_ix)

# reverse
index_to_word = {v:k for k, v in word_to_ix.items()}

WINDOW_SIZE = 5
windows = flatten([list(nltk.ngrams(['<DUMMY>'] * WINDOW_SIZE + c + ['<DUMMY>'] * WINDOW_SIZE, WINDOW_SIZE * 2 + 1)) for c in corpus])

# stores (center word, context word) for skip-5-grams
train_data = []

for window in windows:
  for i in range(WINDOW_SIZE * 2 + 1):
    if window[i] in exclude or window[WINDOW_SIZE] in exclude:
      continue
    if i == WINDOW_SIZE or window[i] == '<DUMMY>':
      continue
    train_data.append((window[WINDOW_SIZE], window[i]))


def prepare_sequence(seq, word_to_ix):
  # print("seq: {}".format(seq))
  idxs = list(map(lambda w: word_to_ix[w] if word_to_ix.get(w) is not None else word_to_ix["<UNK>"], seq))
  return(autograd.Variable(torch.tensor(idxs, dtype = torch.long)))

def prepare_word(word, word_to_ix):
  return(autograd.Variable(torch.tensor([word_to_ix[word]], dtype = torch.long) if word_to_ix.get(word) is not None else torch.tensor(word_to_ix["<UNK>"], dtype = torch.long)))

X_p = []
y_p = []

for tr in train_data:
  X_p.append(prepare_word(tr[0], word_to_ix).view(1, -1))
  y_p.append(prepare_word(tr[1], word_to_ix).view(1, -1))

train_data = list(zip(X_p, y_p))

print(len(train_data))

print(train_data[:3])

## Unigram Distribution for sampling

Z = 0.001

num_total_words = sum([c for w, c in word_count.items() if w not in exclude])

unigram_table = []

for word in vocab:
  # print("New: {}".format(int(((word_count[word]/num_total_words)**0.75)/Z)))
  unigram_table.extend([word] * int(((word_count[word]/num_total_words)**0.75)/Z))

def negative_sampling(targets, unigram_table, k):
  batch_size = targets.size(0)
  neg_samples = []
  for i in range(batch_size):
    nsample = []
    target_index = targets[i].data.tolist()[0]
    while len(nsample) < k:
      neg = random.choice(unigram_table)
      if word_to_ix[neg] == target_index:
        continue
      nsample.append(neg)
    neg_samples.append(prepare_sequence(nsample, word_to_ix).view(1, -1))
  return(torch.cat(neg_samples))

print(len(word_to_ix))

class SGNS(nn.Module):
  
  def __init__(self, vocab_size, ndim):
    super(SGNS, self).__init__()
    self.embedding_v = nn.Embedding(vocab_size, ndim) # Input Word embedding
    self.embedding_u = nn.Embedding(vocab_size, ndim) # Output Word Embedding
    self.logsigmoid = nn.LogSigmoid()

    # Xavier Initialization
    initrange = (2.0/ (vocab_size + ndim))**5
    self.embedding_v.weight.data.uniform_(-initrange, initrange)
    self.embedding_u.weight.data.uniform_(-0.0, 0.0)

  def forward(self, center_words, target_words, negative_words):
    center_vectors = self.embedding_v(center_words)
    target_vectors = self.embedding_u(target_words)
    neg_vectors = -self.embedding_u(negative_words)

    positive_score = target_vectors.bmm(center_vectors.transpose(1, 2)).squeeze(2)
    negative_score = torch.sum(neg_vectors.bmm(center_vectors.transpose(1, 2)).squeeze(2), 1).view(neg_vectors.size(0), -1)

    loss = self.logsigmoid(positive_score) + self.logsigmoid(negative_score)

    return(-torch.mean(loss))

  def predict(self, inputs):
    vectors = self.embedding_v(inputs)
    return(vectors)
  
## Training

EMBEDDING_SIZE = 30
BATCH_SIZE = 256
EPOCH = 100
NEG = 10

losses = []
model = SGNS(len(word_to_ix), EMBEDDING_SIZE)

optimizer = optim.Adam(model.parameters(), lr = 0.001)

for epoch in range(EPOCH):
  for i, batch in enumerate(get_batch(BATCH_SIZE, train_data)):
    inputs, targets = zip(*batch)
    # Input = center word, Target = Context Word to be predicted

    inputs = torch.cat(inputs)
    targets = torch.cat(targets)
    neg_samples = negative_sampling(targets, unigram_table, NEG)
    model.zero_grad()

    loss = model(inputs, targets, neg_samples)

    loss.backward()
    optimizer.step()
    # print('Loss: {}'.format(loss.data.tolist()))
    losses.append(loss.data.tolist())

  if epoch % 10 == 0:
    print("Epoch : %d, mean_loss = %0.2f" % (epoch, np.mean(losses)))
    losses = [] # reset loss for next epoch

def word_similarity(target, vocab, n = 10):
  target_vector = model.predict(prepare_word(target, word_to_ix))

  similarities = []
  for i in range(len(vocab)):
    if vocab[i] == target:
      continue
    vector = model.predict(prepare_word(list(vocab)[i], word_to_ix))
    
    cosine_sim = F.cosine_similarity(target_vector, vector).data.tolist()[0]
    similarities.append([vocab[i], cosine_sim])
  return(sorted(similarities, key = lambda x: x[i], reverse = True)[:n])

test = 'man'

print("Similarity for {}:\n".format(test) + word_similarity(test, vocab))





