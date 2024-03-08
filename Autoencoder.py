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

def validate_cyl(net, test_loader):
    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #evaluate on test set with average absolute error across all test set
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            images = batch[0].to(device)
            images = images.flatten(start_dim=1)
            reconst, _ = net(images)
            test_loss += torch.mean(torch.abs((reconst-images))).item()
    return test_loss/len(test_loader)

def train_cyl(net, opt, train_loader, test_loader, epochs = 100, error = nn.L1Loss(), iters_cycle = 100):
    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Reproducibility  
    torch.manual_seed(0);

    #save losses
    losses = []

    #train
    iters = 0
    for epoch in range(epochs):
        #iterate through dataloader
        for batch in train_loader:
            #separate batch into labels and images
            images = batch[0].to(device)
            
            #make predictions
            reconst, latent = net(images)
            
            #calculate loss
            loss = error(reconst, images.flatten(start_dim=1))
            
            #backpropagate gradients with Adam algorithm, this is the magic of pytorch and autograd
            loss.backward()
            opt.step()
            
            #reset gradients
            net.zero_grad()
            
            #save losses
            losses.append(loss.item())
            
            #log progress
            if iters%iters_cycle==0:    
                print('Epoch: {}/{}     Iter: {}     Loss: {}'.format(epoch, epochs, iters, loss.item()))
            iters +=1

    # outputs    
    out = {}
    out['optimizer'] = type (opt).__name__
    out['losses'] = losses
    out['net'] = net
    out['test_error'] = validate_cyl(net, test_loader)

    return out

def plot_cyl_reconst(out_fname, net, dataloader_test, Nx, Ny, grid_x, grid_y):
    ### Plot Results ###
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ## plot reconstruction
    sample = next(iter(dataloader_test))  #grab the next batch from the dataloader
    reconst, latent = net(sample[0].to(device))#[0].detach().cpu()
    reconst = reconst.view(-1, Ny, Nx).detach().cpu()

    fig, axs = plt.subplots(3,1, figsize=(8, 11))

    # Plot sample
    sample_plot = axs[0].pcolormesh(grid_x, grid_y, sample[0][0], cmap='RdBu', shading='auto', vmax=2, vmin=-2)
    circle = plt.Circle((0, 0), 0.5, color='grey')
    axs[0].add_patch(circle)
    axs[0].set_title('Test Sample')
    axs[0].set_ylabel(r'$y$')
    # axs[0].set_xlabel(r'$x$')
    axs[0].set_aspect('auto')

    # Plot reconstruction
    reconst_plot = axs[1].pcolormesh(grid_x, grid_y, reconst[0], cmap='RdBu', shading='auto', vmax=2, vmin=-2)
    circle = plt.Circle((0, 0), 0.5, color='grey')
    axs[1].add_patch(circle)
    axs[1].set_title('Reconstruction')
    axs[1].set_ylabel(r'$y$')
    # axs[1].set_xlabel(r'$x$')
    axs[1].set_aspect('auto')

    # Calculate error
    error = reconst[0]-sample[0][0]
    MAE = torch.mean(torch.abs(error)).item()

    # Plot error
    #error_plot = axs[2].pcolormesh(grid_x, grid_y, error, vmin = 1e-3,cmap='plasma', shading='auto', norm = 'log')
    error_plot = axs[2].pcolormesh(grid_x, grid_y, error, vmax=2, vmin=-2, cmap='RdBu', shading='auto')
    circle = plt.Circle((0, 0), 0.5, color='grey')
    axs[2].add_patch(circle)
    axs[2].set_title('Error, MAE = {:.2}'.format(MAE))
    axs[2].set_ylabel(r'$y$')
    axs[2].set_xlabel(r'$x$')
    axs[2].set_aspect('auto')

    # Add colorbar
    fig.colorbar(sample_plot, ax=axs[0], pad = 0.01, label = r'$\omega$')
    fig.colorbar(reconst_plot, ax=axs[1], pad = 0.01, label = r'$\omega$')
    fig.colorbar(error_plot, ax=axs[2], pad = 0.01, label = r'$\omega$')

    plt.tight_layout()
    plt.savefig(out_fname)
    plt.close(fig)

def validate_mnist(net, test_loader):
    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #evaluate on test set with average absolute error across all test set
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            images = batch[0].to(device)
            reconst, _ = net(images)
            test_loss += torch.mean((reconst - images)**2).item()
    return test_loss/len(test_loader)

def train_mnist(net, opt, train_loader, test_loader, epochs = 100, error = nn.MSELoss(), iters_cycle = 1000):
    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Reproducibility  
    torch.manual_seed(0);

    #save losses
    losses = []

    #train
    iters = 0
    for epoch in range(epochs):
        #iterate through dataloader
        for batch in train_loader:
            #separate batch into labels and images
            images = batch[0].to(device)
            
            #make predictions
            reconst, latent = net(images)
            
            #calculate loss
            loss = error(reconst, images)
            
            #backpropagate gradients with Adam algorithm, this is the magic of pytorch and autograd
            loss.backward()
            opt.step()
            
            #reset gradients
            net.zero_grad()
            
            #save losses
            losses.append(loss.item())
            
            #log progress
            if iters%iters_cycle==0:    
                print('Epoch: {}/{}     Iter: {}     Loss: {}'.format(epoch, epochs, iters, loss.item()))
            iters +=1
    
    #outputs
    out = {}
    out['optimizer'] = type (opt).__name__
    out['losses'] = losses
    out['net'] = net
    out['test_error'] = validate_mnist(net, test_loader)

    return out

def plot_mnist_reconst(out_fname, net, dataloader_test):
    ### Plot Results ###
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ## plot reconstruction
    sample = next(iter(dataloader_test))  #grab the next batch from the dataloader
    reconst, _ = net(sample[0].to(device))

    fig, axs = plt.subplots(3,1, figsize=(4,10))

    # Plot sample
    sample_plot = axs[0].imshow(sample[0][0,0], cmap='gray')
    axs[0].set_title('Test Sample')
    axs[0].set_aspect('auto')

    # Plot reconstruction
    reconst_plot = axs[1].imshow(reconst[0,0].detach().cpu(), cmap='gray')
    axs[1].set_title('Reconstruction')
    axs[1].set_aspect('auto')

    # Calculate error
    error = (reconst[0,0].detach().cpu()-sample[0][0,0])**2
    MSE = torch.mean(error**2).item()

    # Plot error
    error_plot = axs[2].imshow(error, cmap='gray')
    axs[2].set_title('Squared Error, MSE = {:.2}'.format(MSE))
    axs[2].set_aspect('auto')

    # Add colorbar
    fig.colorbar(sample_plot, ax=axs[0], pad = 0.01, label = r'$\omega$')
    fig.colorbar(reconst_plot, ax=axs[1], pad = 0.01, label = r'$\omega$')
    fig.colorbar(error_plot, ax=axs[2], pad = 0.01, label = r'$\omega$')

    plt.tight_layout()
    plt.savefig(out_fname)
    plt.close(fig)