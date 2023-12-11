import random
import transformers
import torch
import numpy as np

def init_seed(s):
  transformers.set_seed(s)
  random.seed(s)
  np.random.seed(s)
  torch.manual_seed(s)