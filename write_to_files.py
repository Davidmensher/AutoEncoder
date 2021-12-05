import matplotlib.pyplot as plt
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



def write_parameters(batch_size, lr, epochs, train_loss, val_loss, model, name, total_train_loss, total_val_loss):
    print("train losses = ", total_train_loss)
    print("validation losses = ", total_val_loss)
    #calculating layers of encoder
    enc_l = [module for module in model.encoder.modules() if isinstance(module, nn.Sequential)]
    enoder_layers = enc_l[0]
    
    #calculating layers of decoder
    dec_l = [module for module in model.decoder.modules() if isinstance(module, nn.Sequential)]
    decoder_layers = dec_l[0]
    
    conv_l = [module for module in model.encoder.modules() if isinstance(module, nn.Conv2d)]
    num_layers = 2 * len(conv_l)
    
    
    f = open('{name}/param_file.txt'.format(name = name), "w")
    L = ["----------Parameters file----------\n",
         "Batch size = {Batch_size} \n".format(Batch_size = batch_size),
         "train losses {0} \n".format(total_train_loss),
         "val losses {0} \n".format(total_val_loss),
         "Learning rate = {lr}\n".format(lr  = lr),
         "epochs = {epochs}\n".format(epochs= epochs),
         "Final training loss = {train_loss}\n".format(train_loss=train_loss),
         "Final validation loss = {val_loss}\n".format(val_loss=val_loss),
         "Number of convolution layers = {num_layers}\n".format(num_layers=num_layers),
         "Structure of Encoder = {enoder_layers}\n".format(enoder_layers=enoder_layers),
         "Structure of Decoder = {decoder_layers}\n".format(decoder_layers=decoder_layers)]
         
    f.writelines(L)
    f.close()


def plot_graphs(epochs, train_loss, val_loss, name):
    l = [i for i in range(epochs)]

    plt.plot(l, train_loss,'--g', label="train loss")
    plt.plot(l, val_loss, label="vaidation loss")
    plt.title("Losses plot")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.savefig('{name}/Loss_plot.png'.format(name = name))
