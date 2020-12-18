import torch
import torch.nn as nn

from torch.nn import init

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReNetBlock(nn.Module):
    def __init__(self, infil=64, outfil=64):
        super(ReNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(infil, outfil, kernel_size=3, stride=1, padding=1)
        self.bn1 =  nn.InstanceNorm2d(outfil)
        self.relu1 =  nn.ReLU(False)

        self.conv2 = nn.Conv2d(outfil, outfil, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.InstanceNorm2d(outfil)
        self.relu2 =  nn.ReLU(False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu2(out)

        return out


class ConvBNR(nn.Module):
  def __init__(self, in_channels, out_channels, activation=nn.LeakyReLU(0.2, False), kw=3, strides=2, pad=1):
        super(ConvBNR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kw, stride=strides, padding=pad)
        self.bn1 = nn.InstanceNorm2d(out_channels, affine=False)
        self.activation = activation

  def forward(self, x):
        return self.bn1(self.conv1(self.activation(x)))


class TransposeConvBNR(nn.Module):
  def __init__(self, in_channels, out_channels, activation=nn.LeakyReLU(0.2, False), kw=4, strides=2, pad=1):
        super(TransposeConvBNR, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kw, stride=strides, padding=pad)
        self.bn1 = nn.InstanceNorm2d(out_channels, affine=False)
        self.activation = activation

  def forward(self, x):
        return self.bn1(self.conv1(self.activation(x)))


class VAE1_encoder(nn.Module):
    def __init__(self, in_channels=3, n_res_blocks=4, n_layers=5):
        super(VAE1_encoder, self).__init__()
        self.n_layers = n_layers
        self.layer = nn.ModuleList()
        self.layer1 = ConvBNR(in_channels, 64, kw=7, pad=3, strides=1)
        for j in range(n_layers):
            self.layer.append(ConvBNR(64, 64))

        self.res_blks = nn.ModuleList()
        self.n_res_blocks = n_res_blocks
        for j in range(n_res_blocks):
          self.res_blks.append(ReNetBlock(64, 64))

    def forward(self, x):
        x=self.layer1(x)
        for j in range(self.n_layers):
          x= self.layer[j](x)

        for j in range(self.n_res_blocks):
           x=self.res_blks[j](x)
        return x


class VAE1_decoder(nn.Module):
    def __init__(self, in_channels=64, out_channel=3, n_res_blocks=4, n_layers=5):
        super(VAE1_decoder, self).__init__()
        self.n_layers = n_layers
        self.layer = nn.ModuleList()
        for j in range(n_layers):
            self.layer.append(TransposeConvBNR(64, 64, pad=1))

        self.layer1 = nn.Conv2d(64, out_channel, kernel_size=1, stride=1)
        self.res_blks = nn.ModuleList()
        self.n_res_blocks = n_res_blocks
        for j in range(n_res_blocks):
          self.res_blks.append(ReNetBlock(64, 64))

    def forward(self, x):
        for j in range(self.n_res_blocks):
           x = self.res_blks[j](x)
        
        for j in range(self.n_layers):
            x = self.layer[j](x)

        x = self.layer1(x)
        return x


class VAE1(nn.Module):
  def __init__(self, z_dim=256):
        super(VAE1, self).__init__()
        self.n_layers = 2
        self.down_fact = 2**self.n_layers
        down_img_size = 256 // self.down_fact
        self.fdim = (down_img_size)**2 * 64
        self.size=[64, down_img_size, down_img_size]
        self.encoder = VAE1_encoder(n_layers=self.n_layers)
        self.decoder = VAE1_decoder(n_layers=self.n_layers)
        self.fc1 = nn.Linear(self.fdim, z_dim)
        self.fc2 = nn.Linear(self.fdim, z_dim)
        self.fc3 = nn.Linear(z_dim, self.fdim)

  def flat(self, input):
        return input.view(input.size(0), 1, -1)

  def unflat(self, input):
        return input.view(input.size(0), self.size[0], self.size[1], self.size[2])

  def init_weights(self, init_type="xavier", gain=0.02):
      def init_func(m):
          classname = m.__class__.__name__
          if classname.find("BatchNorm2d") != -1:
              if hasattr(m, "weight") and m.weight is not None:
                  init.normal_(m.weight.data, 1.0, gain)
              if hasattr(m, "bias") and m.bias is not None:
                  init.constant_(m.bias.data, 0.0)
          elif hasattr(m, "weight") and (classname.find("Conv") != -1):
              if init_type == "normal":
                  init.normal_(m.weight.data, 0.0, gain)
              elif init_type == "xavier":
                  init.xavier_normal_(m.weight.data, gain=gain)
              elif init_type == "xavier_uniform":
                  init.xavier_uniform_(m.weight.data, gain=1.0)
              elif init_type == "kaiming":
                  init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
              elif init_type == "orthogonal":
                  init.orthogonal_(m.weight.data, gain=gain)
              elif init_type == "none":  # uses pytorch's default init method
                  m.reset_parameters()
              else:
                  raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
              if hasattr(m, "bias") and m.bias is not None:
                  init.constant_(m.bias.data, 0.0)
          elif hasattr(m, "weight") and (classname.find("Linear") != -1):
              init.normal_(m.weight.data,0.0,gain)

      self.apply(init_func)
      for m in self.children():
            if hasattr(m, "init_weights"):
                m.init_weights(init_type, gain)
        
  def reparameterize(self, mu, logvar):
      std = logvar.mul(0.5).exp_()
      esp = torch.randn(*mu.size()).to(device)
      z = mu + std * esp  
      return z
  
  def bottleneck(self, h):
      mu, logvar = self.fc1(h), self.fc2(h)
      logvar = logvar * 1e-3 #lazy way to stabilize the training.
      z = self.reparameterize(mu, logvar)
      return z, mu, logvar

  def encode(self, x):
      h = self.encoder(x)
      h = self.flat(h)
      z, mu, logvar = self.bottleneck(h)
      return z, mu, logvar

  def decode(self, z):
      z = self.fc3(z)
      z_img = self.unflat(z)
      z = self.decoder(z_img)
      return z, z_img

  def forward(self, x, flow='all'):
      z_latent, mu, logvar = self.encode(x)
      z,z_out = self.decode(z_latent)
      if(flow == 'all'):
        return z, mu, logvar,z_out
      else:
        return z, mu, logvar,z_out,z_latent


################################## LSGAN #############################################


class ConvBNRelu(nn.Module):
    def __init__(self, nin,nout,stride=2):
        super(ConvBNRelu, self).__init__()

        self.layer=nn.Sequential(
            nn.Conv2d(nin, nout, kernel_size=4, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.layer(x)


class Discriminator(nn.Module):
    def __init__(self, nChannels=64, ndf=64):
        super(Discriminator, self).__init__()
        self.layer1 = ConvBNRelu(nChannels, ndf, stride=2) 
        self.layer2 =  ConvBNRelu(ndf, ndf*2, stride=2) 
        self.layer3 =  ConvBNRelu(ndf*2, ndf*4, stride=2) 
        self.layer4 =  ConvBNRelu(ndf*4, ndf*8, stride=2) 
        self.layer5 =   ConvBNRelu(ndf*8, 1, stride=1) 
        self.final = nn.Conv2d(1, 1, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out=self.final(out)
        return out