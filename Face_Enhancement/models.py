import torchvision.models as models

#Code pour VGG provenant de https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss

class CondInstNorm(nn.Module):
    def __init__(self,N_in_latent,N_in_degraded_face):
        super().__init__()
        self.norm_layer = nn.InstanceNorm2d(N_in_latent, affine=False)
        nhidden = 128
        kw=3;
        self.mlp = nn.Sequential(nn.Conv2d(N_in_degraded_face, nhidden, kernel_size=kw, padding=kw//2), nn.ReLU())
        self.gamma = nn.Conv2d(nhidden, N_in_latent, kernel_size=kw, padding=kw//2)
        self.beta = nn.Conv2d(nhidden, N_in_latent, kernel_size=kw, padding=kw//2)

    def forward(self, x, featmap_degraded): # is conv ouptut from z and featmap_degraded is computed from the degraded face out. 
        inst_normalized = self.norm_layer(x)
        featmap_degraded = functionnal.interpolate(featmap_degraded, size=x.size()[2:], mode="nearest")
        ml_out = self.mlp(featmap_degraded)
        gamma = self.gamma(ml_out)
        beta = self.beta(ml_out)
        out = inst_normalized * (1 + gamma) + beta
        return out

class ResnetBlock(nn.Module):
    def __init__(self, dim, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            nn.Conv2d(dim, dim, kernel_size=kernel_size),
            nn.BatchNorm2d(dim),
            activation,
            nn.ReflectionPad2d(pw),
            nn.Conv2d(dim, dim, kernel_size=kernel_size),
            nn.BatchNorm2d(dim),

        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out



class Generator(nn.Module):
  def __init__(self):
    super().__init__()
    input_nc = 3
    ngf = 64
    resnet_initial_kernel_size = 7
    resnet_n_downsample = 4
    resnet_n_blocks = 3
    resnet_kernel_size  = 3
    output_nc = 3
    
    self.init = nn.Sequential(
                  nn.ReflectionPad2d(resnet_initial_kernel_size // 2),
                  nn.Conv2d(input_nc, ngf, kernel_size=resnet_initial_kernel_size, padding=0),
                  nn.BatchNorm2d(ngf),
                  nn.ReLU(False)
                )


    self.down1 =  nn.Sequential(
                    nn.Conv2d(ngf , ngf * 2, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(False)
                  )
    
    self.down2 =  nn.Sequential(
                    nn.Conv2d(ngf *2, ngf * 4, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(ngf * 4),
                    nn.ReLU(False)
                   )
    
    self.down3 =  nn.Sequential(
                    nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(ngf * 8),
                    nn.ReLU(False)
                  )
    
    self.down4 =  nn.Sequential(
                    nn.Conv2d(ngf * 8, ngf * 16, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(ngf * 16),
                    nn.ReLU(False)
                  )
    

    self.resnet = nn.Sequential(
        ResnetBlock(ngf * 16, activation=nn.ReLU(False), kernel_size=resnet_kernel_size,),
        ResnetBlock(ngf * 16, activation=nn.ReLU(False), kernel_size=resnet_kernel_size,),
        ResnetBlock(ngf * 16, activation=nn.ReLU(False), kernel_size=resnet_kernel_size,),
    )

    self.up1= nn.Sequential(
        nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(ngf * 8),
        nn.ReLU(False),
    )

    self.up2= nn.Sequential(
        nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(False),
    )

    self.up3= nn.Sequential(

        nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(False),
    )

    self.up4= nn.Sequential(
        nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(ngf),
        nn.ReLU(False),
    )

    self.fin= nn.Sequential(
        nn.ReflectionPad2d(3),
        nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
        nn.Tanh(),
    )

    self.model = nn.Sequential(
        nn.ReflectionPad2d(resnet_initial_kernel_size // 2),
        nn.Conv2d(input_nc, ngf, kernel_size=resnet_initial_kernel_size, padding=0),
        nn.BatchNorm2d(ngf),
        nn.ReLU(False),

        nn.Conv2d(ngf , ngf * 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(False),

        nn.Conv2d(ngf *2, ngf * 4, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(False),

        nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(ngf * 8),
        nn.ReLU(False),

        nn.Conv2d(ngf * 8, ngf * 16, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(ngf * 16),
        nn.ReLU(False),

        ResnetBlock(ngf * 16, activation=nn.ReLU(False), kernel_size=resnet_kernel_size,),
        ResnetBlock(ngf * 16, activation=nn.ReLU(False), kernel_size=resnet_kernel_size,),
        ResnetBlock(ngf * 16, activation=nn.ReLU(False), kernel_size=resnet_kernel_size,),

        nn.ConvTranspose2d(ngf * 16, ngf * 8, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(ngf * 8),
        nn.ReLU(False),

        nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(False),

        nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(False),

        nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(ngf),
        nn.ReLU(False),

        nn.ReflectionPad2d(3),
        nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
        nn.Tanh(),
    )

    self.cbn1 = CondInstNorm(128,3)
       
    self.cbn2 = CondInstNorm(256,3)
    self.cbn3 = CondInstNorm(512,3)
    self.cbn4 = CondInstNorm(1024,3)


  def forward(self, x):

    init = self.init(x)
    down1 = self.down1(init)
    test1 = self.cbn1(down1,x)

    down2 = self.down2(down1)

    test2 = self.cbn2(down2,x)
  

    down3 = self.down3(down2)
    test3 = self.cbn3(down3,x)

    down4 = self.down4(down3)
    test4 = self.cbn4(down4,x)

    resnet = self.resnet(down4)

    up1 = self.up1(resnet+test4)

    up2 = self.up2(up1+test3)
    up3 = self.up3(up2+test2)
    up4 = self.up4(up3+test1)
    output = self.fin(up4)



    return output


class Discriminator(nn.Module):
  def __init__(self,nChannels=3, ndf=64):
    super().__init__()
    kw = 3
    pw = int(np.ceil((kw - 1.0) / 2))

    self.conv = nn.Sequential(
        nn.Conv2d(3, ndf, kw, stride=2, padding=pw),
        nn.BatchNorm2d(ndf),

        nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw),
        nn.BatchNorm2d(ndf*2),

        nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw),
        nn.BatchNorm2d(ndf*4),

        nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw),
        nn.BatchNorm2d(ndf*8),

        nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw),
        nn.BatchNorm2d(ndf*8),
        nn.LeakyReLU(0.2, False)
    )

    

  def forward(self, x):
      if x.size(2) != 256 or x.size(3) != 256:
        x = F.interpolate(x, size=(256, 256), mode="bilinear")

      x = self.conv(x)
      x = x.view(x.size(0), -1)

      return x