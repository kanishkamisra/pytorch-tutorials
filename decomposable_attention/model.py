import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_align(sims, embeddings):
  assert(sims.size()[1] == embeddings.size()[0])
  softmax = nn.Softmax(dim = 1)
  sims = softmax(sims)
  reweighted = torch.matmul(sims, embeddings)
  return(reweighted)

class Decomp(nn.Module):
  def __init__(self, **kwargs):
    super(Decomp, self).__init__()
    self.emb_size = kwargs["EMB_SIZE"]
    self.hidden_size = kwargs["HIDDEN_SIZE"]
    self.output_size = kwargs["OUTPUT_SIZE"]
    self.vocab = kwargs["VOCAB"]

    # Layers
    self.embed = nn.Embedding(self.vocab, self.emb_size)
    self.linear_transform = nn.Sequential(nn.Linear(self.emb_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, self.hidden_size))
    self.linear_pair = nn.Sequential(nn.Linear(self.emb_size*2, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, self.hidden_size))
    self.linear_decide = nn.Sequential(nn.Linear(self.hidden_size*2, self.hidden_size), nn.ReLU(),
    nn.Linear(self.hidden_size, self.output_size), nn.LogSoftmax(dim = 0))

  def forward(self, sentence1, sentence2):
    sent1, sent2 = [self.embed(x) for x in [sentence1, sentence2]]

    transformed1, transformed2 = [self.linear_transform(x) for x in [sent1, sent2]]

    sim1 = torch.mm(transformed1, torch.t(transformed2))
    sim2 = torch.mm(transformed2, torch.t(transformed1))

    concatenated1 = torch.cat((sent1, get_align(sim1, sent2)), 1)
    concatenated2 = torch.cat((sent2, get_align(sim2, sent1)), 1)

    paired1, paired2 = [self.linear_pair(x) for x in [concatenated1, concatenated2]]

    v1, v2 = [torch.sum(x, 0) for x in [paired1, paired2]]

    pred = self.linear_decide(torch.cat((v1, v2), 0))

    return(pred)
