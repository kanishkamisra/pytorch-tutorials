import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class decomp(nn.Module):
  def __init__(self, **kwargs):
    super(MLP, self).__init__()
    self.emb_size = kwargs["EMB_SIZE"]
    self.hidden_size = kwargs["HIDDEN_SIZE"]
