''' Autoencoder objects '''

import torch
import torch.nn as nn
import torchvision
import numpy as np 
import matplotlib.pyplot as plt
from collections import OrderedDict

### Helper modules ###

# Encoder as a fully connected network
class Encoder(nn.Module):
    def __init__(self, im_size, n_latent, n_hidden, NN_width, taper = False):
        super(Encoder, self).__init__()
        if not taper:
            #create input layer
            self.l1 = nn.Linear(im_size, NN_width)
            self.a1 = nn.LeakyReLU(0.2, inplace=True)

            #hidden layers
            layers = []
            for i in range(n_hidden):
                layers.append(('hidden{}'.format(i+1),nn.Linear(NN_width, NN_width)))
                #layers.append(('batch norm{}'.format(i+1),nn.BatchNorm1d(NN_width)))
                layers.append(('ReLU{}'.format(i+1), nn.LeakyReLU(0.2, inplace=True)))

            self.hidden = nn.Sequential(OrderedDict(layers))

            #output layer
            self.out = nn.Linear(NN_width,n_latent)
        else:
            taper_amount = 2 # divide layer size by 2 each time
            NN_widths = [min(n_latent*(taper_amount**i), NN_width) for i in range(n_hidden)]
            NN_widths = NN_widths[::-1] #reverse for simplicity

            #create input layer
            self.l1 = nn.Linear(im_size, NN_widths[0])
            self.a1 = nn.LeakyReLU(0.2, inplace=True) 

            layers = []
            for i in range(n_hidden-1):
                layers.append(('hidden{}'.format(i+1),nn.Linear(NN_widths[i], NN_widths[i+1])))
                #layers.append(('batch norm{}'.format(i+1),nn.BatchNorm1d(NN_width)))
                layers.append(('ReLU{}'.format(i+1), nn.LeakyReLU(0.2, inplace=True)))

            self.hidden = nn.Sequential(OrderedDict(layers))

            #output layer
            self.out = nn.Linear(NN_widths[-1],n_latent)

    #forward pass
    def forward(self, x):
        x = x.view(x.size()[0],-1) #flatten the images into a vector
        x = self.l1(x)
        x = self.a1(x)
        x = self.hidden(x)
        return self.out(x)

# Decoder as a fully connected network
class Decoder(nn.Module):
    def __init__(self, im_size, n_latent, n_hidden, NN_width, taper = False, square = True):
        super(Decoder, self).__init__()

        self.im_size = im_size
        self.square = square #if output is square (MNIST), else just returns a vector for you to reshape

        #create hidden layers
        if not taper:
            #create input layer
            self.l1 = nn.Linear(n_latent, NN_width)
            self.a1 = nn.LeakyReLU(0.2, inplace=True)

            layers = []
            for i in range(n_hidden):
                layers.append(('hidden{}'.format(i+1),nn.Linear(NN_width, NN_width)))
                #layers.append(('batch norm{}'.format(i+1),nn.BatchNorm1d(NN_width)))
                layers.append(('ReLU{}'.format(i+1), nn.LeakyReLU(0.2, inplace=True)))

            self.hidden = nn.Sequential(OrderedDict(layers))

            #output layer
            self.out = nn.Linear(NN_width,im_size)
        else:
            taper_amount = 2
            NN_widths = [min(n_latent*(taper_amount**i), NN_width) for i in range(n_hidden)]

            #create input layer
            self.l1 = nn.Linear(n_latent, NN_widths[0])
            self.a1 = nn.LeakyReLU(0.2, inplace=True)

            layers = []
            for i in range(n_hidden-1):
                layers.append(('hidden{}'.format(i+1),nn.Linear(NN_widths[i], NN_widths[i+1])))
                #layers.append(('batch norm{}'.format(i+1),nn.BatchNorm1d(NN_width)))
                layers.append(('ReLU{}'.format(i+1), nn.LeakyReLU(0.2, inplace=True)))

            self.hidden = nn.Sequential(OrderedDict(layers))

            #output layer
            self.out = nn.Linear(NN_widths[-1],im_size)

    #forward pass
    def forward(self, x):
        x = self.l1(x)
        x = self.a1(x)
        x = self.hidden(x)
        x = self.out(x)
        if self.square:
            im_size = int(np.sqrt(self.im_size))
            return x.view(x.size()[0],1,im_size,im_size)
        else:
            return x
    

### Trainable Autoencoder class ###
class AutoEncoder(nn.Module):
    def __init__(self, im_size, n_latent, n_hidden, NN_width, taper = False, square = True):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(im_size, n_latent, n_hidden, NN_width, taper = taper)
        self.decoder = Decoder(im_size, n_latent, n_hidden, NN_width, taper = taper, square = square)

    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)
        return x, latent