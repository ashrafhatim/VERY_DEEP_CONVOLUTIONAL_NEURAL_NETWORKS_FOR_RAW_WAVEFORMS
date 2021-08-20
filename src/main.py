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

from models import *
import argparse


def test_model(model: torch.nn.Module, train_loader):
  """
  Test Loop
  """

  loss_function = torch.nn.CrossEntropyLoss()
  # optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)

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
      correct += ( torch.argmax(y,dim=1).item() == t.item() )

    average_loss = total_loss / len(train_loader)
    accuracy = (correct / len(train_loader)) * 100
    
  return average_loss, accuracy


def train_model(model_class, epochs = 1, exp_name = None, val_fold = 9, savePath = None, dataPath = None, metadataPath = None):
  """
  Train Loop
  """
  # models = []

  samples = 0
  train_loaders = []

  val_loader =  datasets("/content/UrbanSound8K/audio", "/content/UrbanSound8K/metadata/UrbanSound8K.csv", str(val_fold))
  for fold in range(1,10):
    if fold == val_fold: continue
    loader = datasets("/content/UrbanSound8K/audio", "/content/UrbanSound8K/metadata/UrbanSound8K.csv", str(fold))
    train_loaders.append( loader )
    samples += len( loader )

  model = model_class()
  model.apply(glorot_init)

  # intialise summary writer
  loss_sw = SummaryWriter(os.path.join(savePath, exp_name, "tensorboard"))
  # accuracy_sw = SummaryWriter(os.path.join(savePath, expName, "tensorboard"))

  loss_function = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)

  model.to(device)
  model.train()

  for epoch in tqdm.tqdm(range(epochs)):
    total_loss = 0
    correct = 0
    
    for loader in train_loaders:
      for i, (x, t) in enumerate(loader):

        optimizer.zero_grad()

        x = x.to(device)
        t = t.to(device)
        y = model(x)

        loss = loss_function(y, t)
        loss.backward()
        optimizer.step()

        total_loss +=  float(loss)
        # print('plus ', float(loss), " total loss = ", total_loss)
        correct += ( torch.argmax(y,dim=1).item() == t.item() )

    scheduler.step()


    train_loss = total_loss / samples
    train_accuracy = (correct / samples) * 100
    val_loss, val_accuracy = test_model(model, val_loader)
    print("epo {} loss => ".format(epoch), train_loss )

    loss_sw.add_scalars("loss for exp({})-val_fold({})".format(exp_name, val_fold), {"train":train_loss, "val":val_loss}, epoch)
    loss_sw.add_scalars("accuracy for exp({})-val_fold({})".format(exp_name, val_fold), {"train":train_accuracy, "val":val_accuracy}, epoch)
      
  # models.append(model)

  PATH = os.path.join(savePath,  exp_name + "_val_fold_" + str(val_fold) + ".ckpt") #"/content/drive/MyDrive/MODELS/"
  torch.save({
              'model_state_dict': model.state_dict(),
              }, PATH ) #+ exp_name + "_val_fold_" +str(val_fold))


  return model





if __name__=="__main__":
    parser = argparse.ArgumentParser(description='deep conv')


    parser.add_argument('--epochs', default=1, type=int, help='number of total epochs to run')

    parser.add_argument('--val-fold', default=1, type=int, help='number of total epochs to run')

    parser.add_argument('--exp-name', default="M3", type=str, help='number of total epochs to run')

    parser.add_argument('--savePath', default="", type=str, help='number of total epochs to run')

    parser.add_argument('--dataPath', default="", type=str, help='number of total epochs to run')

    parser.add_argument('--metadataPath', default="", type=str, help='number of total epochs to run')

   


    args=parser.parse_args()

    
    if args.exp_name == "M3":
        model_class = M3
    elif args.exp_name == "M5":
        model_class = M5
    elif args.exp_name == "M11":
        model_class = M11
    elif args.exp_name == "M18":
        model_class = M18
    elif args.exp_name == "M34_res":
        model_class = M34_res
        
    model = train_model(model_class, epochs = args.epochs, exp_name = args.exp_name, val_fold = args.val_fold, savePath=args.savePath, dataPath= args.dataPath, metadataPath= args.metadataPath)

    train_loader =  datasets(args.dataPath, args.metadataPath, str(10))
    average_loss, accuracy = test_model(model, train_loader)

    print("Test loss = {} - Test accuracy = {}".format(average_loss, accuracy))


