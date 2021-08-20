import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import Audio, display # display(Audio(audio_path, rate=8000))
import tqdm
import glob
import time
import os

from torch.utils.tensorboard import SummaryWriter


# M3 (0.2M) 

class M3(nn.Module):
  def __init__(self):
    super(M3, self).__init__()

    """
    M3 modle 
    """
    
    # input 1 * 32000
    self.conv1 = nn.Conv1d(1, 256, 80, 4)
    self.bn1 = nn.BatchNorm1d(256)
    self.maxpool1 = nn.MaxPool1d(4)
    self.conv2 = nn.Conv1d(256, 256, 3)
    self.bn2 = nn.BatchNorm1d(256)
    self.maxpool2 = nn.MaxPool1d(4)

    self.averpool = nn.AdaptiveAvgPool1d(1)
    self.fc = nn.Linear(256, 10)
  
  def forward(self, x):
    x = x.unsqueeze(0)
    x = self.conv1(x)
    x = self.bn1(x)

    x = self.maxpool1(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.maxpool2(x)
    x = self.averpool(x)
    x = x.reshape(-1, 256)
    x = self.fc(x)

    return x

# M5 (0.5M) 

class M5(nn.Module):
  def __init__(self):
    super(M5, self).__init__()
    
    """
    M5 model
    """
    # input 1 * 32000
    self.conv1 = nn.Conv1d(1, 128, 80, 4)
    self.bn1 = nn.BatchNorm1d(128)
    self.maxpool1 = nn.MaxPool1d(4)

    self.convList = []

    dims = [128, 128, "pool", 128, 256, "pool", 256, 512, "pool"]
    for i in range(len(dims) - 1):
      if dims[i+1] == "pool":
        self.convList.append(nn.MaxPool1d(4))
        continue
      if dims[i] == "pool":
        continue
      self.convList.append(nn.Conv1d(dims[i], dims[i+1], 3))
      self.convList.append(nn.BatchNorm1d(dims[i+1]))

    self.seq = nn.Sequential(*self.convList)
    self.averpool = nn.AdaptiveAvgPool1d(1)
    self.fc = nn.Linear(512, 10)
  
  def forward(self, x):
    x = x.unsqueeze(0)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.maxpool1(x)

    # for i in range(len(self.convList)):
    #   x = self.convList[i](x)
    x = self.seq(x)

    x = self.averpool(x)
    x = x.reshape(-1, 512)
    x = self.fc(x)

    return x

# M11 (1.8M) 

class M11(nn.Module):
  def __init__(self):
    super(M11, self).__init__()

    """
    M11 model
    """
    
    # input 1 * 32000
    self.conv1 = nn.Conv1d(1, 64, 80, 4)
    self.bn1 = nn.BatchNorm1d(64)
    self.maxpool1 = nn.MaxPool1d(4)

    self.convList = []

    dims = [64, 64, 64, "pool", 64, 128, 128, "pool", 128, 256, 256, 256, "pool", 256, 512, 512]
    for i in range(len(dims) - 1):
      if dims[i+1] == "pool":
        self.convList.append(nn.MaxPool1d(4))
        continue
      if dims[i] == "pool":
        continue
      self.convList.append(nn.Conv1d(dims[i], dims[i+1], 3))
      self.convList.append(nn.BatchNorm1d(dims[i+1]))

    self.seq = nn.Sequential(*self.convList)
    self.averpool = nn.AdaptiveAvgPool1d(1)
    self.fc = nn.Linear(512, 10)
  
  def forward(self, x):
    x = x.unsqueeze(0)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.maxpool1(x)

    # for i in range(len(self.convList)):
    #   x = self.convList[i](x)

    x = self.seq(x)
    
      
    x = self.averpool(x)
    x = x.reshape(-1, 512)
    x = self.fc(x)

    return x

# M18 (3.7M) 

class M18(nn.Module):
  def __init__(self):
    super(M18, self).__init__()
    
    """
    M18 model
    """

    # input 1 * 32000
    self.conv1 = nn.Conv1d(1, 64, 80, 4)
    self.bn1 = nn.BatchNorm1d(64)
    self.maxpool1 = nn.MaxPool1d(4)

    self.convList = []

    dims = [64, 64, 64, 64, 64, "pool", 64, 128, 128, 128, 128, "pool", 128, 256, 256, 256, 256, "pool", 256, 512, 512, 512, 512]
    for i in range(len(dims) - 1):
      if dims[i+1] == "pool":
        self.convList.append(nn.MaxPool1d(4))
        continue
      if dims[i] == "pool":
        continue
      self.convList.append(nn.Conv1d(dims[i], dims[i+1], 3))
      self.convList.append(nn.BatchNorm1d(dims[i+1]))

    self.seq = nn.Sequential(*self.convList)
    self.averpool = nn.AdaptiveAvgPool1d(1)
    self.fc = nn.Linear(512, 10)
  
  def forward(self, x):
    x = x.unsqueeze(0)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.maxpool1(x)

    # for i in range(len(self.convList)):
    #   x = self.convList[i](x)

    x = self.seq(x)
    x = self.averpool(x)
    x = x.reshape(-1, 512)
    x = self.fc(x)

    return x

# M34_res (4M)

class M34_res(nn.Module):
  def __init__(self):
    super(M34_res, self).__init__()

    """
    M34_res model
    """
    
    # input 1 * 32000
    self.conv1 = nn.Conv1d(1, 48, 80, 4)
    self.bn1 = nn.BatchNorm1d(48)
    self.maxpool1 = nn.MaxPool1d(4)

    #first residual block
    #1
    self.res2 = self.residual_net(48, 48)
    self.bn2 = nn.BatchNorm1d(48)
    self.relu2 = nn.ReLU()
    #2
    self.res3 = self.residual_net(48, 48)
    self.bn3 = nn.BatchNorm1d(48)
    self.relu3 = nn.ReLU()
    #3
    self.res4 = self.residual_net(48, 48)
    self.bn4 = nn.BatchNorm1d(48)
    self.relu4 = nn.ReLU()

    self.maxpool4 = nn.MaxPool1d(4)

    #second residual block
    #1
    self.res5 = self.residual_net(48, 96)
    self.bn5 = nn.BatchNorm1d(96)
    self.relu5 = nn.ReLU()
    #2
    self.res6 = self.residual_net(96, 96)
    self.bn6 = nn.BatchNorm1d(96)
    self.relu6 = nn.ReLU()
    #3
    self.res7 = self.residual_net(96, 96)
    self.bn7 = nn.BatchNorm1d(96)
    self.relu7 = nn.ReLU()
    #4
    self.res8 = self.residual_net(96, 96)
    self.bn8 = nn.BatchNorm1d(96)
    self.relu8 = nn.ReLU()
    
    self.maxpool8 = nn.MaxPool1d(4)

    #third residual block
    #1
    self.res9 = self.residual_net(96, 192)
    self.bn9 = nn.BatchNorm1d(192)
    self.relu9 = nn.ReLU()
    #2
    self.res10 = self.residual_net(192, 192)
    self.bn10 = nn.BatchNorm1d(192)
    self.relu10 = nn.ReLU()
    #3
    self.res11 = self.residual_net(192, 192)
    self.bn11 = nn.BatchNorm1d(192)
    self.relu11 = nn.ReLU()
    #4
    self.res12 = self.residual_net(192, 192)
    self.bn12 = nn.BatchNorm1d(192)
    self.relu12 = nn.ReLU()
    #5
    self.res13 = self.residual_net(192, 192)
    self.bn13 = nn.BatchNorm1d(192)
    self.relu13 = nn.ReLU()
    #6
    self.res14 = self.residual_net(192, 192)
    self.bn14 = nn.BatchNorm1d(192)
    self.relu14 = nn.ReLU()
    
    self.maxpool14 = nn.MaxPool1d(4)

    #last residual block
    #1
    self.res15 = self.residual_net(192, 384)
    self.bn15 = nn.BatchNorm1d(384)
    self.relu15 = nn.ReLU() 
    #2
    self.res16 = self.residual_net(384, 384)
    self.bn16 = nn.BatchNorm1d(384)
    self.relu16 = nn.ReLU()
    #3
    self.res17= self.residual_net(384, 384)
    self.bn17 = nn.BatchNorm1d(384)
    self.relu17 = nn.ReLU()

    self.averpool = nn.AdaptiveAvgPool1d(1)
    self.fc = nn.Linear(384, 10)
    
  def residual_net(self, input, out):
    return nn.Sequential(
        nn.Conv1d(input, out, 3, padding=1),
        nn.BatchNorm1d(out),
        nn.ReLU(),
        nn.Conv1d(out, out, 3, padding=1),
        nn.BatchNorm1d(out)
         )
  def forward(self, x):
    x = x.unsqueeze(0)
    x = self.conv1(x)
    x = self.bn1(x)

    # first residual block
    #1
    residual = x
    x = self.res2(x)
    x += residual
    x = self.bn2(x)
    x = self.relu2(x)
    #2
    residual = x
    x = self.res3(x)
    x += residual
    x = self.bn3(x)
    x = self.relu3(x)
    #3
    residual = x
    x = self.res4(x)
    x += residual
    x = self.bn4(x)
    x = self.relu4(x)

    x = self.maxpool4(x)

    # second residual block
    #1
    residual = x
    x = self.res5(x)
    residual = nn.Upsample(size=(x.shape[1],x.shape[2]), mode='nearest')(residual.unsqueeze(0)).reshape(-1, x.shape[1], x.shape[2])
    x += residual
    x = self.bn5(x)
    x = self.relu5(x)
    #2
    residual = x
    x = self.res6(x)
    x += residual
    x = self.bn6(x)
    x = self.relu6(x)
    #3
    residual = x
    x = self.res7(x)
    x += residual
    x = self.bn7(x)
    x = self.relu7(x)
    #4
    residual = x
    x = self.res8(x)
    x += residual
    x = self.bn8(x)
    x = self.relu8(x)

    x = self.maxpool8(x)

    #third residual block
    #1
    residual = x
    x = self.res9(x)
    residual = nn.Upsample(size=(x.shape[1],x.shape[2]), mode='nearest')(residual.unsqueeze(0)).reshape(-1, x.shape[1], x.shape[2])
    x += residual
    x = self.bn9(x)
    x = self.relu9(x)
    #2
    residual = x
    x = self.res10(x)
    x += residual
    x = self.bn10(x)
    x = self.relu10(x)
    #3
    residual = x
    x = self.res11(x)
    x += residual
    x = self.bn11(x)
    x = self.relu11(x)
    #4
    residual = x
    x = self.res12(x)
    x += residual
    x = self.bn12(x)
    x = self.relu12(x)
    #5
    residual = x
    x = self.res13(x)
    x += residual
    x = self.bn13(x)
    x = self.relu13(x)
    #6
    residual = x
    x = self.res14(x)
    x += residual
    x = self.bn14(x)
    x = self.relu14(x)

    x = self.maxpool14(x)

    # last residual block
    #1
    x = self.res15(x)
    residual = nn.Upsample(size=(x.shape[1],x.shape[2]), mode='nearest')(residual.unsqueeze(0)).reshape(-1, x.shape[1], x.shape[2])
    x += residual
    x = self.bn15(x)
    x = self.relu15(x)
    #2
    x = self.res16(x)
    x += residual
    x = self.bn16(x)
    x = self.relu16(x)
    #3
    x = self.res17(x)
    x += residual
    x = self.bn17(x)
    x = self.relu17(x)

    x = self.averpool(x)
    x = x.reshape(-1, 384)
    x = self.fc(x)

    return x

def glorot_init(layer):
    """
    glorot initialisation for the wieght
    """
    if type(layer) == nn.Conv1d or type(layer)==nn.Linear:
        nn.init.xavier_uniform_(layer.weight.data)
