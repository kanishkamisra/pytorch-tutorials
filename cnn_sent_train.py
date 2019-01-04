from collections import defaultdict
from cnn_sent import CNN
from sklearn.utils import shuffle
from torch.autograd import Variable

import torch
import torch.optim as optim
import torch.nn as nn

def train(data, params):
  model = CNN(**params)

  # extracts all parameters that are needed for backprop.
  parameters = filter(lambda p: p.requires_grad, model.parameters())
  optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
  loss = nn.CrossEntropyLoss()

  
  