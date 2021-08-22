import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np

import glob
import os


# import EarlyStopping
# from pytorchtools import EarlyStopping


class datasets(Dataset):
  def __init__(self, dataset_path, metadata_path, folds):

    """
    dataset data class
    ---
    dataset path: str
    metadata_path: str
    folds: int, list[int]
    """
    self.folds = folds
    self.audios = glob.glob(dataset_path + "/fold" + str(self.folds) + "/*")
    self.metadata = pd.read_csv(metadata_path)

  def __len__(self):
    return len(self.audios)
  
  def __getitem__(self, idx):
    audios, freq = torchaudio.load(self.audios[idx], normalize=True)

    # downsample form 44100hz to 8000hz -> divide by five
    # the input vector length must be 32000 as written in the paper
    # to keep the dimention of the input and apply the downsampling -> pad the input to the size of (32000 * 5 = 160000) before downsampling

    audios = audios.mean(0, keepdim=True)
    n,m = audios.shape
    padding = (160000 - m) // 2
    audios = F.pad(audios, (padding, padding), 'constant', 0)
    audios = audios[:, ::5]

    # get the label
    audio_names = self.audios[idx].split('/')[-1]
    labels = self.metadata.loc[self.metadata["slice_file_name"] == audio_names]["classID"].item()
 
    return audios, labels