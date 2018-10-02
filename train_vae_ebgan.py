import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import autograd
#from torchvision.datasets import MNIST
#from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F

torch.manual_seed(123)


def plot_loss(loss_array,name):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(loss_array)
    plt.savefig('loss_'+name)


def weights_init_G(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def weights_init_D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.002)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def loss_function(recon_x, x, mu, logvar):
    MSE = (recon_x -x).pow(2).mean()
    #BCE = F.binary_cross_entropy(recon_x.squeeze(), x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    #KLD = torch.sum(KLD_element).mul_(-0.5)
    KLD = KLD_element.mul_(-0.5).mean()
    batch_size = recon_x.size(0)


    return MSE+KLD, MSE, KLD


def run_trainer(train_loader, netE, netG, netD, args):

    margin = args.margin

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_D, betas=(0.5,0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_G,betas=(0.5,0.999))
    optimizerE = optim.Adam(netE.parameters(), lr = args.lr_G)

    noise = torch.FloatTensor(args.batch_size, args.n_z, 1, 1)
    noise = noise.cuda()
    noise = Variable(noise)
    

    netE.apply(weights_init_G)
    netG.apply(weights_init_G)
    netD.apply(weights_init_D)


    criterion = nn.MSELoss()

    for epoch in range(1000):
        G_loss_epoch = 0
        recon_loss_epoch = 0
        kld_loss_epoch = 0
        D_loss_epoch = 0
        features_loss_epoch = 0
        errD_real_epoch = 0
        errD_fake_epoch = 0

        data_iter = iter(train_loader)
        i = 0

        for i, (images, labels) in enumerate(train_loader):
     
            images = Variable(images)
            images = images.cuda()


            #train netD
            for p in netD.parameters():
                p.requires_grad = True
            
            for p in netG.parameters():
                p.requires_grad = False

            for p in netE.parameters():
                p.requires_grad = False


            netD.zero_grad()

            #train real
            output, _ = netD(images)
            errD_real = (output-images).pow(2).mean()

            errD_real.backward(retain_graph=True)

            #train fake
            noise = noise.data.normal_(0,1)
            fake, _ = netG(noise)
            output, _ = netD(fake)
            errD_fake = (output-fake).pow(2).mean()
            errD_fake = margin - errD_fake
            errD_fake = errD_fake.clamp(min=0)

            errD_fake.backward()

            errD = (errD_real + errD_fake)/2.0
            D_loss_epoch += errD.data.cpu().item()
            errD_real_epoch += errD_real.data.cpu().item()
            errD_fake_epoch += errD_fake.data.cpu().item()
            optimizerD.step()

            #train netE, netG
            for p in netE.parameters():
                p.requires_grad = True
            
            for p in netG.parameters():
                p.requires_grad = True

            for p in netD.parameters():
                p.requires_grad = False

            netE.zero_grad()
            netG.zero_grad()
            
            mu, logvar, z_enc = netE(images)
            recon, _ = netG(z_enc)

            D_fake, features_fake = netD(recon)
            D_real, features_real = netD(images)

            loss, recon_loss, KLD = loss_function(recon, images.squeeze(), mu, logvar)
            features_loss = (features_fake-features_real).pow(2).mean()
            

            recon_loss_epoch += recon_loss.data.cpu().item()
            kld_loss_epoch += KLD.data.cpu().item()
            G_loss_epoch += loss.data.cpu().item()
            features_loss_epoch += features_loss.data.cpu().item()

            features_loss.backward(retain_graph=True)
            KLD.backward(retain_graph=True)
            optimizerG.step()
            optimizerE.step()


            #print('recon.size()', recon.size())
            recon = recon.unsqueeze(1)



            if  i % 200 == 0 :
                print('saving images for batch', i)
                save_image(recon[0:6].data.cpu().detach(), './fake.png')
                save_image(images[0:6].data.cpu().detach(), './real.png')


        #recon_loss_array.append(recon_loss_epoch)
        #d_loss_array.append(d_loss_epoch)

        #if(epoch%5==0):
        #    print('plotting losses')
        #    plot_loss(recon_loss_array,'recon')
        #    plot_loss(d_loss_array, 'disc')

        if(epoch % 1 == 0):
            print("Epoch, features_loss, recon_loss, KLD_loss, D_loss" 
                  ,epoch + 1, features_loss_epoch, recon_loss_epoch, kld_loss_epoch, D_loss_epoch)
