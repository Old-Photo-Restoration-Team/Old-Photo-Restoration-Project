import numpy as np
import torch.nn as nn
import torch
import tqdm

from matplotlib import pyplot as plt

from data import RealPhotosDataset
from utils import display_images
from model import .


torch.manual_seed(1)
torch.cuda.manual_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

learning_rate = 2e-4
batch_size = 16
epochs = 100

################################# Data Generation #####################################
data_set = RealPhotosDataset(path)

train_len = int(len(data_set) * 0.9)
train_set, test_set = random_split(data_set, [train_len, len(data_set) - train_len])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=5)
test_loader = DataLoader(test_set, batch_size=4, num_workers=5)
#######################################################################################

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate, betas=(0.5, 0.999)
)

model=VAE2().to(device)
l1_loss=torch.nn.L1Loss()
means, logvars = [], []

for epoch in range(0, epochs + 1):

    # Training
    if epoch > 0:  # test untrained net first
        model.train()

        train_loss = 0
        kl_loss = 0
        norm1_loss = 0

        with tqdm.tqdm(train_loader, total=int(len(train_loader)), unit="batch", position=0, leave=True) as tepoch:
          for x in tepoch:
              tepoch.set_description(f"Epoch {epoch}")
              x = x.to(device)
              x_hat, mu, logvar,z1 = model(x)

              loss1 = l1_loss(x_hat, x)
              loss2 = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar, 1))
              loss = loss1 + loss2
              kl_loss += loss2.item()
              norm1_loss += loss1.item()
              train_loss += loss.item()

              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
              tepoch.set_postfix(loss=train_loss / len(train_loader.dataset),
                                 l1_loss=norm1_loss/ len(train_loader.dataset), 
                                 kl_loss=kl_loss/ len(train_loader.dataset))
              
        print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')
    
    # Testing
    with torch.no_grad():
        model.eval()
        test_loss = 0.0

        for ix, x in enumerate(test_loader):
            x = x.to(device)
            x_hat, mu, logvar, _ = model(x)
            loss = 0.0
            loss += l1_loss(x_hat, x)
            loss+= torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar, 1))
            test_loss+=loss.item()

            if (ix<1 and epoch % 5) == 0:
              plt.figure(figsize=(10, 4))
              for i in range(4):
                  plt.subplot(1,8, i+1)
                  plt.imshow(np.clip(x_hat.detach().cpu().numpy()[i, :, :, :].transpose(1, 2, 0)+0.5, 0.0, 1.0))
                  plt.axis('off')

              for i in range(4):
                  plt.subplot(1, 8, i + 1 + 4)
                  plt.imshow(np.clip(x.detach().cpu().numpy()[i, :, :].transpose(1, 2, 0)+0.5, 0.0, 1.0))
                  plt.axis('off')
              plt.show()

    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')
    #display_images(x, x_hat, 1)
