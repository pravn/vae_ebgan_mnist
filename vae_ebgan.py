import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.datasets import MNIST
#from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.n_channel = args.n_channel
        self.n_z = args.n_z
        self.n_x = args.n_x
        self.n_y = args.n_y
        self.input_size = args.n_x * args.n_y
        self.dec_neurons = 800

        self.first = nn.Linear(self.n_z,self.dec_neurons)
        self.second = nn.Linear(self.dec_neurons,self.dec_neurons)
        self.third = nn.Linear(self.dec_neurons,self.input_size)
        #self.fourth = nn.Linear(800,800)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU(0.2,inplace=1)

    def forward(self, x):
        x = x.view(-1, self.n_z)
        x = self.relu(self.first(x))
        features = self.relu(self.second(x))
        x = self.relu(self.third(x))
        #x = self.leaky_relu(self.fourth(x))
        x = self.tanh(x)
        return x.view(-1,self.n_x,self.n_y), features


class Encoder(nn.Module):
    def __init__(self,args):
        super(Encoder, self).__init__()
        self.n_channel = args.n_channel
        self.n_z = args.n_z
        self.input_size = args.n_x * args.n_y
        self.enc_neurons = 256

        self.first = nn.Linear(self.input_size, self.enc_neurons)
        self.second = nn.Linear(self.enc_neurons,self.n_z)
        self.third = nn.Linear(self.enc_neurons,self.n_z)

        self.relu = nn.ReLU(True)
        self.leaky_relu = nn.LeakyReLU(0.2,inplace=True)

    def forward(self,x):
        x = x.view(-1,self.input_size)
        #x = self.relu(self.fc(x))
        x = self.leaky_relu(self.first(x))
        x = self.leaky_relu(self.second(x))
        #x = self.leaky_relu(self.third(x))
        return x


class VAE_Encoder(nn.Module):
    def __init__(self,args):
        super(VAE_Encoder, self).__init__()
        self.n_channel = args.n_channel
        self.n_z = args.n_z
        self.input_size = args.n_x * args.n_y

        self.encoder = Encoder(args)
        self.enc_mu = Encoder(args)
        self.enc_logvar = Encoder(args)
        self.decoder = Decoder(args)
        
        self.relu = nn.ReLU(True)
        self.leaky_relu = nn.LeakyReLU(0.2,inplace=True)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self,x):
        x = x.view(-1,self.input_size)
        mu = self.enc_mu(x)
        logvar = self.enc_logvar(x)
        #mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        #recon = self.decoder(z)
        return mu, logvar, z
        
class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        
        self.n_channel = args.n_channel
        self.n_z = args.n_z
        self.n_x = args.n_x
        self.n_y = args.n_y 
        self.input_size = args.n_x * args.n_y
        
        self.encoder = Encoder(args)
        #self.decoder = Decoder(args)
        self.decoder = Decoder(args)
        self.relu = nn.ReLU(True)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.encoder(x)
        x = x.squeeze()
        x, features = self.decoder(x)
        #x = self.tanh(x)
        return  x.view(-1,self.n_x,self.n_y), features




def return_model(args):
    encoder = VAE_Encoder(args)
    decoder = Decoder(args)
    disc = Discriminator(args)

    encoder = encoder.cuda()
    decoder = decoder.cuda()
    disc = disc.cuda()

    print('return model - encoder.cuda(), decoder.cuda(), disc.cuda()')

    return encoder, decoder, disc
