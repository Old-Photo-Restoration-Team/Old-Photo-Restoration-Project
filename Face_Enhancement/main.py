import torch
import torch.nn as nn
import torch.nn.functional as functionnal
from torchvision import transforms
import torch.nn.intrinsic as intr
import torchvision
import matplotlib.pyplot as plt
import os
import requests
from io import BytesIO
import PIL.Image
import tqdm
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = 'cpu'

from data import CreateDataSet
import models

##DATA GENERATION##########
import zipfile
from torch.utils.data import Dataset, DataLoader, random_split
from google.colab import drive
drive.mount('/content/drive')
if not os.path.isdir('targetdir/'):
  with zipfile.ZipFile("drive/MyDrive/DataIA/FFHQSynthPhotos.zip","r") as zip_ref:
      zip_ref.extractall("targetdir")

  with zipfile.ZipFile("drive/MyDrive/DataIA/archive.zip","r") as zip_ref:
      zip_ref.extractall("targetdir")

real_path = "/content/targetdir/archive"
x_path = "/content/targetdir/FFHQSynthPhotos"

batch_size = 16

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
class Clamp(object):
    def __init__(self, min, max):
        self.min = float(min)
        self.max = float(max)

    def __call__(self, x):
        new_x = torch.clamp(x, self.min, self.max)
        
        return new_x

class Permute(object):
    def __init__(self, dims):
        self.dims = dims

    def __call__(self, x):
        new_x = x.permute(*self.dims)
        
        return new_x

preprocessing = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=IMAGENET_MEAN,
                                                         std=IMAGENET_STD),
                                   ])

mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)
std = torch.tensor(IMAGENET_STD, dtype=torch.float32)

postprocessing = transforms.Compose([
                                     transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()),
                                     Clamp(0, 1),
                                    ])



data_set = CreateDataSet(real_path,x_path)

data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False,num_workers=5)
##############

## training ####
epochs = 100
netG = Generator().to(device)
vgg = VGGPerceptualLoss().to(device)

netD = Discriminator().to(device)

optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))


img_list = []
G_losses = []
D_losses = []
iters = 0

for epoch in range(0, epochs + 1):
    if epoch > 0: 
        netG.train()
        netD.train();
        train_lossG = 0.;
        train_lossD = 0.;


        for samp_real,samp_x in tqdm.tqdm(data_loader, total=int(len(data_loader)),position=0, leave=True):
            samp_real = samp_real.to(device)
            samp_x = samp_x.to(device)

            optimizerG.zero_grad()

            if(samp_x.size()[0] == 16):
              fake_image = netG(samp_x)


              discriminator_out_fakeG = netD(fake_image)

              GLoss = vgg(samp_real,fake_image) +torch.mean((1-discriminator_out_fakeG)**2)
              GLoss.backward()
              optimizerG.step()

              optimizerD.zero_grad()

              with torch.no_grad():
                fake_image = netG(samp_x)
                fake_image = fake_image.detach()


              discriminator_out_fake = netD(fake_image)
              discriminator_out_real = netD(samp_real)

              DLoss = vgg(fake_image, samp_real) +torch.mean((1-discriminator_out_real)**2) + torch.mean((discriminator_out_fake)**2)
              DLoss.backward()
              optimizerD.step()

            
        with torch.no_grad():
          netG.eval()

          for samp_real,samp_x in data_loader:
              samp_x = samp_x.to(device)
              sample_fake = netG(samp_x)
              fig=plt.figure()
              plt.imshow(np.transpose(postprocessing(sample_fake[0]).detach().cpu().numpy(),(1,2,0)))
              break;
           
###########