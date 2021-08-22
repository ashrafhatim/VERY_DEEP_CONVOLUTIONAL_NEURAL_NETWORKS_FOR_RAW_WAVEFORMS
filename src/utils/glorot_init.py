import torch.nn as nn

def glorot_init(layer):
  """
  glorot initialisation for the wieght
  """
  if type(layer) == nn.Conv1d or type(layer)==nn.Linear:
    nn.init.xavier_uniform_(layer.weight.data)