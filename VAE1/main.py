import numpy as np
import torch.nn as nn
import torch
import tqdm

from torch.utils.data import random_split
from matplotlib import pyplot as plt

from data import FakeandRealOldPhotosGenerator
from utils import display_images
from model import .


torch.manual_seed(1)
torch.cuda.manual_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learning_rate = 1e-4
batch_size = 16
epochs = 100

################################# Data Generation #####################################
data_set = FakeAndRealOldPhotosGenerator(real_path, x_path)

train_len = int(len(data_set)*0.9)
train_set, test_set = random_split(data_set, [train_len, len(data_set) - train_len])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=5)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=5)
#######################################################################################

netG = VAE1().to(device)
netD = Discriminator(nChannels=64, ndf=64).to(device)
l1_loss=torch.nn.L1Loss()

optimizerD = torch.optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))

means, logvars, costs = [], [], []

for epoch in range(0, epochs + 1):

    # Training
    if epoch > 0:  # test untrained net first
        netG.train()
        netD.train()
        train_lossG = 0.
        train_lossD = 0.

        for  samp_real,samp_x in train_loader:

            samp_real = samp_real.to(device)
            samp_x = samp_x.to(device)

            x_hat_real, mu_real, logvar_real, z_real = netG(samp_real)
            x_hat_x, mu_x, logvar_x, z_x = netG(samp_x)

            out_r = netD(z_real.detach())
            out_x = netD(z_x.detach())
            lossD = 0.5 *(torch.mean((1.0 - out_r)**2) + torch.mean((out_x)**2))
            train_lossD += lossD.item()

            optimizerD.zero_grad()
            lossD.backward()
            optimizerD.step()

            loss_real_l1 = l1_loss(x_hat_real, samp_x) * 256.0
            loss_real_kl = 0.5 * torch.mean(logvar_real.exp() - logvar_real - 1.0 + mu_real**2)
            loss_real = loss_real_l1 + loss_real_kl

            loss_x_l1 = l1_loss(x_hat_x, samp_x) * 256.0
            loss_x_kl = 0.5 * torch.mean(logvar_x.exp() - logvar_x - 1.0 + mu_x**2)
            loss_x = loss_x_l1 + loss_x_kl   

            out_r = netD(z_real)
            out_x = netD(z_x)
            lossGAN = 0.5 * (torch.mean(out_r**2) + torch.mean((1.0 - out_x)**2))

            lossF= loss_real + loss_x + lossGAN # Total loss
            train_lossG += lossF.item()

            optimizerG.zero_grad()
            lossF.backward()
            optimizerG.step()
            costs.append([loss_real.item(), loss_x.item(), lossGAN.item(), lossD.item()])

        print(f'====> Epoch: {epoch} Average discriminator loss: {train_lossD / len(train_loader.dataset):.4f} Generator loss: {train_lossG / len(train_loader.dataset):.4f}')
    
    with torch.no_grad():
        netG.eval()
        test_loss_r = 0
        test_loss_x = 0
        for ix,(samp_real, samp_x) in enumerate(test_loader):
            samp_real = samp_real.to(device)
            samp_x = samp_x.to(device)

            x_hat_real, mu_real, logvar_real, z_real= netG(samp_real)
            x_hat_x, mu_x, logvar_x, z_x= netG(samp_x)

            loss_r=l1_loss(samp_real, x_hat_real)
            loss_x=l1_loss(samp_x, x_hat_x)
            test_loss_r += loss_r.item()
            test_loss_x += loss_x.item()

            if(ix<1 and epoch%5 == 0):
              plt.figure(figsize=(10, 8))

              for i in range(4):
                  plt.subplot(1, 8, i+1)
                  plt.imshow(np.clip(x_hat_real.detach().cpu().numpy()[i, : ,:, :].transpose(1, 2, 0)+0.5, 0.0, 1.0))
                  plt.axis('off')

              for i in range(4):
                  plt.subplot(1, 8, i+1+4)
                  plt.imshow(np.clip(samp_real.detach().cpu().numpy()[i, :, :].transpose(1, 2, 0)+0.5, 0.0, 1.0))
                  plt.axis('off')
              plt.show()

              plt.figure(figsize=(10, 8))
              for i in range(4):
                  plt.subplot(1, 8, i+1)
                  plt.imshow(np.clip(x_hat_x.detach().cpu().numpy()[i, :, :, :].transpose(1, 2, 0)+0.5, 0.0, 1.0))
                  plt.axis('off')

              for i in range(4):
                  plt.subplot(1, 8, i+1+4)
                  plt.imshow(np.clip(samp_x.detach().cpu().numpy()[i, :, :].transpose(1, 2, 0)+0.5, 0.0, 1.0))
                  plt.axis('off')
              plt.show()

    test_loss_r /= len(test_loader.dataset)
    test_loss_x /= len(test_loader.dataset)
    print(f'====> Test set loss_r: {test_loss_r:.4f} test loss_x: {test_loss_x:.4f}')
    # display_images(x, x_hat, 1)
