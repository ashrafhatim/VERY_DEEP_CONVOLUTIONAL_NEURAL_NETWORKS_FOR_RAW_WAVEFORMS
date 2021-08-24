import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import Audio, display # display(Audio(audio_path, rate=8000))
# import tqdm
import glob
# import time
import os

from torch.utils.tensorboard import SummaryWriter # tensorboard
# from keras.callbacks import EarlyStopping

import argparse

from data.urbansound8k import datasets
from utils import glorot_init
from models import *

def test_model(model: torch.nn.Module, train_loader):
  """
  Test Loop
  """

  loss_function = torch.nn.CrossEntropyLoss()

  model.to(device)
  model.eval()

  for epoch in range(1):
    
    total_loss = 0
    correct = 0
    
    for i, (x, t) in enumerate(train_loader):

      x = x.to(device)
      t = t.to(device)
      y = model(x)

      loss = loss_function(y, t)
      total_loss += float(loss)
      correct += ( torch.argmax(y,dim=1) == t ).sum().item()

    average_loss = total_loss / len(train_loader.dataset)
    accuracy = (correct / len(train_loader.dataset)) * 100

  model.train()
  return average_loss, accuracy

def train_model(model_class, epochs = 1, exp_name = None, val_fold = 10):

  """
  Train Loop
  """

  best_loss = 1000
  no_increase = 0
  
  kw = {'num_workers': 8, 'pin_memory': True} if device == 'cuda' else {} 

  folds = []
  for fold in range(1,11):
    if fold == val_fold: continue
    folds.append(fold)

  train_data = datasets(path + "audio", path + "metadata/UrbanSound8K.csv", folds )
  train_loader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True, **kw)

  val_data = datasets(path + "audio", path + "metadata/UrbanSound8K.csv", 10 )
  val_loader = torch.utils.data.DataLoader(val_data, batch_size = 64, shuffle = True, **kw)

  model = model_class()
  model.apply(glorot_init.glorot_init)

  # intialise summary writer
  sw = SummaryWriter(savePath + exp_name + "_val_fold_" +str(val_fold) + "_tensorboard")
  
  # loss function and scheduler
  loss_function = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)

  model.to(device)
  model.train()

  for epoch in range(epochs):
    total_loss = 0
    correct = 0
    
    for x, t in train_loader:

      optimizer.zero_grad()

      x = x.to(device)
      t = t.to(device)
      y = model(x)

      loss = loss_function(y, t)
      loss.backward()
      optimizer.step()

      with torch.no_grad():
        total_loss +=  float(loss)
      correct += ( torch.argmax(y,dim=1) == t ).sum().item()
    scheduler.step()

    train_loss = total_loss / len( train_loader.dataset)
    train_accuracy = (correct / len( train_loader.dataset)) * 100
    with torch.no_grad():
      val_loss, val_accuracy = test_model(model, val_loader)
    print("epo {} - train_loss: {} - train_accuracy: {}".format(epoch, train_loss, train_accuracy))
    print("epo {} - val_loss: {} - val_accuracy: {}".format(epoch, val_loss, val_accuracy))

    sw.add_scalars("loss for exp({})-val_fold({})".format(exp_name, val_fold), {"train":train_loss, "val":val_loss}, epoch)
    sw.add_scalars("accuracy for exp({})-val_fold({})".format(exp_name, val_fold), {"train":train_accuracy, "val":val_accuracy}, epoch)

    if val_loss < best_loss:
      no_increase = 0
      best_loss = val_loss
      # save the best model
      torch.save({
              'model_state_dict': model.state_dict(),
              }, savePath + exp_name + "_val_fold_" +str(val_fold) + "_best_model"+".ckpt")
    else:
      no_increase += 1
      if no_increase > 10:
        print("early stop !")
        break

  torch.save({
              'model_state_dict': model.state_dict(),
              }, savePath + exp_name + "_val_fold_" +str(val_fold) + ".ckpt")


  return model


if __name__=="__main__":

    parser = argparse.ArgumentParser()
      
    parser.add_argument("--exp-name", default= "M3", type= str)
    parser.add_argument("--epochs", default= 5, type= int)
    parser.add_argument("--val-fold", default= 10, type= int)
    parser.add_argument("--dataset-path", default="/content/UrbanSound8K/", type= str)
    parser.add_argument("--save-path", default="/content/drive/MyDrive/MODELS/", type= str)

    args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
path = args.dataset_path # dataset folder
savePath = args.save_path # save folder
model_class =  M3 if args.exp_name=="M3" else M5 if args.exp_name=="M5" else M11 if args.exp_name=="M11" else M18 if args.exp_name== "M18" else M34_res if args.exp_name=="M34_res" else None

train_model(model_class=model_class, epochs=args.epochs, exp_name=args.exp_name, val_fold=args.val_fold)


