import numpy as np
import torch
import os
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image, make_grid
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from write_to_files import write_parameters, plot_graphs



def run(bs,name, lr, loss_f):
    ##### parameters
    batch_size = bs
    exp_name = name
    lr = lr
    
    tf = T.Compose([
        # Resize to constant spatial dimensions
        T.Resize((256, 256)),
        # PIL.Image -> torch.Tensor
        T.ToTensor(),
        # Dynamic range [0,1] -> [-1, 1]
        T.Normalize(mean=(.5,.5,.5), std=(.5,.5,.5)),
    ])

    # Loading the data
    ds_gwb = ImageFolder("../dataset", tf)
    split_lengths = [39000,1000]
    ds_train, ds_test = random_split(ds_gwb, split_lengths)
    train_dl = DataLoader(ds_train, batch_size, shuffle=True)
    val_dl  = DataLoader(ds_test,  batch_size, shuffle=True)
    
    
    
    # Model class
    class AE(torch.nn.Module):
        def __init__(self):
            super().__init__()
              
            self.encoder = nn.Sequential(
              # conv 1
              nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=2),
              nn.BatchNorm2d(8),
              nn.LeakyReLU(),
              nn.MaxPool2d(kernel_size=2, stride=2),
  
              # conv 2
              nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2),
              nn.BatchNorm2d(16),
              nn.LeakyReLU(),
              nn.MaxPool2d(kernel_size=2, stride=2),
  
              # conv 3
              nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
              nn.BatchNorm2d(32),
              nn.LeakyReLU(),
              nn.MaxPool2d(kernel_size=2, stride=2),
              
              nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
              nn.BatchNorm2d(64),
              nn.LeakyReLU(),
              nn.MaxPool2d(kernel_size=2, stride=2),
  
              # conv 4
              nn.Conv2d(in_channels=64, out_channels=4, kernel_size=5, stride=1, padding=2),
              nn.BatchNorm2d(4),
              nn.LeakyReLU(),
              nn.MaxPool2d(kernel_size=2, stride=2)
              
            )

              
            self.decoder = nn.Sequential(
              # conv 6
              nn.Upsample(scale_factor=2, mode='bilinear'),
              nn.ConvTranspose2d(in_channels=4, out_channels=64, kernel_size=5, stride=1, padding=2),
              nn.BatchNorm2d(64),
              nn.LeakyReLU(),
  
              # conv 7
              nn.Upsample(scale_factor=2, mode='bilinear'),
              nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2),
              nn.BatchNorm2d(32),
              nn.LeakyReLU(),
              
  
              # conv 8
              nn.Upsample(scale_factor=2, mode='bilinear'),
              nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2),
              nn.BatchNorm2d(16),
              nn.LeakyReLU(),
              
              nn.Upsample(scale_factor=2, mode='bilinear'),
              nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=5, stride=1, padding=2),
              nn.BatchNorm2d(8),
              nn.LeakyReLU(),
  
              # conv 9
              nn.Upsample(scale_factor=2, mode='bilinear'),
              nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=5, stride=1, padding=2),
              nn.BatchNorm2d(3),
              nn.LeakyReLU()

        )

        # forward calculatuoin
        def forward(self, x):
            encoded = self.encoder(x)
            print(encoded.size())
            decoded = self.decoder(encoded)
            return decoded


    model = AE()
    
    
    loss_fn1 = loss_f
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = lr,
                                 weight_decay = 1e-8)

                                 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
            
    # validatoin loss calculation
    def calc_loss(model):
        with torch.no_grad():
          val_loss = 0
          for (images, _) in val_dl:
              images = images.to(device)
              reconstructed = model(images)
              loss = loss_fn1(reconstructed, images) 
              val_loss += loss.item()
          return val_loss/len(val_dl)       
        
    # training loop
    jj = 0
    epochs = 20
    total_train_loss = []
    total_val_loss = []
    for epoch in range(epochs):
        train_loss = []
        print("------epoch {0} ---------".format(epoch))
        for (images, _) in train_dl:
            #images = images.reshape(-1, 256*256)
            images = images.to(device)
            reconstructed = model(images)
            a = reconstructed
            if jj%1000 == 0:
              save_image(a, '{name}/img{num}.png'.format(name = exp_name, num = jj//1000))
            jj = jj +1
            loss = loss_fn1(reconstructed, images) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(abs(loss.item())/2)
        train_loss = sum(train_loss)/len(train_dl)
        total_train_loss.append(train_loss)
    
        val_loss = calc_loss(model)
        total_val_loss.append(val_loss)
        print("--- train loss {train_loss}, val loss {val_loss}".format(train_loss = train_loss, val_loss = val_loss))
        print(total_train_loss)
        print(total_val_loss)
        
    save_image(a, '{name}/img{num}.png'.format(name = exp_name, num = jj//1000))
    write_parameters(batch_size, lr, epochs, total_train_loss[epochs-1], total_val_loss[epoch-1], model, exp_name, total_train_loss, total_val_loss)
    plot_graphs(epochs, total_train_loss, total_val_loss, exp_name)


run(64,"exp14_david", 1e-5, nn.L1Loss())
