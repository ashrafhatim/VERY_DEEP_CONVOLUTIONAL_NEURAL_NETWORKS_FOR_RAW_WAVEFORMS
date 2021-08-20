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


class datasets(Dataset):
  def __init__(self, dataset_path, metadata_path, fold):

    """
    dataset data loader
    """

    self.audios = glob.glob(dataset_path + "/fold" + fold + "/*")
    self.metadata = pd.read_csv(metadata_path)
    self.fold = fold

  def __len__(self):
    return len(self.audios)
  
  def __getitem__(self, idx):
    audio, freq = torchaudio.load(self.audios[idx], normalize=True)

    # downsample form 44100hz to 8000hz -> divide by five
    # the input vector length must be 32000 as written in the paper
    # to keep the dimention of the input and apply the downsampling -> pad the input to the size of (32000 * 5 = 160000) before downsampling

    audio = audio.mean(0, keepdim=True)
    n,m = audio.shape
    padding = (160000 - m) // 2
    audio = F.pad(audio, (padding, padding), 'constant', 0)
    audio = audio[:, ::5]

    # get the label
    audio_name = self.audios[idx].split('/')[-1]
    label = self.metadata.loc[self.metadata["slice_file_name"] == audio_name]["classID"].item()
    label = torch.tensor([label])
 
    return audio, label