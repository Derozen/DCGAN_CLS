import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
#from utils import Concat_embed
import pdb


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.nz = 100
        self.txt_size = 1024
        self.nt = 128
        self.nc = 3
        self.ngf = 128

        self.fcg = nn.Sequential(
            nn.Linear(self.txt_size,self.nt),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.netG1 = nn.Sequential(
            nn.ConvTranspose2d(self.nt + self.nz, self.ngf * 8, 4, 1, 0),
            nn.BatchNorm2d(self.ngf * 8))
        self.netG2 = nn.Sequential(
            #nn.ReLU(True),
            nn.Conv2d(self.ngf * 8, self.ngf * 2, 1, 1,0),
			nn.BatchNorm2d(self.ngf * 2),
			nn.ReLU(True),
            nn.Conv2d(self.ngf * 2, self.ngf * 2, 3, 1, 1),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(self.ngf * 2, self.ngf * 8, 3, 1, 1),
            nn.BatchNorm2d(self.ngf * 8))
        self.netG3 = nn.Sequential(
            nn.ReLU(True),
            #state size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(self.ngf*4))
        self.netG4 = nn.Sequential(
            nn.Conv2d(self.ngf * 4, self.ngf, 1, 1, 0),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf, 3, 1, 1),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            nn.Conv2d(self.ngf, self.ngf * 4, 3, 1, 1),
            nn.BatchNorm2d(self.ngf * 4))
        self.netG5 = nn.Sequential(
            nn.ReLU(True),
            #state size: (self.ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            #state size: (self.ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            #state size: (self.ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1),
            nn.Tanh() 
        )
        #self. embed = Concat_embed(c_dim, z_dim)
        

    def forward(self, t, z):
        #z_c = self.embed(z,c)
        phi_t = self.fcg(t).unsqueeze(2).unsqueeze(3)
        conc_vec = torch.cat([z, phi_t], dim=1)
        y1 = self.netG1(conc_vec)
        y2 = self.netG2(y1)
        y3 = y1 + y2
        y4 = self.netG3(y3)
        y5 = self.netG4(y4)
        y6 = y4 + y5
        output = self.netG5(y6)

        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.nc = 3
        self.ndf = 64
        self.txt_size = 1024
        self.nt = 128

        self.convD1 = nn.Sequential(
            #-- input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            #-- state size: (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2,True),
            #-- state size: (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(self.ndf * 4),
            #-- state size: (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1),
            nn.BatchNorm2d(self.ndf * 8))
        self.convD2 = nn.Sequential(
            #-- state size: (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, self.ndf * 2, 1, 1, 0),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(self.ndf * 2, self.ndf * 2, 3, 1, 1),
            nn.BatchNorm2d(self.ndf *2),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(self.ndf * 2, self.ndf * 8, 3, 1, 1),
            nn.BatchNorm2d(self.ndf * 8))
        self.RL = nn.LeakyReLU(0.2,True)
        self.fcD = nn.Sequential(
            nn.Linear(self.txt_size,self.nt),
            nn.BatchNorm1d(self.nt),
            nn.LeakyReLU(0.2,True),
        )
        self.netD4 = nn.Sequential(
            nn.Conv2d(self.ndf * 8 + self.nt, self.ndf * 8, 1, 1),
            nn.BatchNorm2d(self.ndf*8),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(self.ndf * 8, 1, 4, 4),
            nn.Sigmoid()
        )
    def forward(self, im, t):
        t_1 = self.fcD(t).unsqueeze(2).repeat(1, 1, 4).unsqueeze(3).repeat(1,1,1,4)
        y = self.convD1(im)
        y1 = self.convD2(y)
        y = y + y1
        y = self.RL(y)
        # print(y.shape)
        # print(t_1.shape)
        conc_vec = torch.cat([y, t_1], 1)
        output = self.netD4(conc_vec)
        return output.reshape(output.shape[0], 1), y
    




class Rnetwork(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.batch_size = opt.batchSize
        self.num_caption = opt.numCaption
        self.txt_size = opt.txtSize
        self.replicate = opt.replicate

    def forward(self, x):
        if self.replicate == 1:
            # (batchSize, txtSize)
            x = x.view(
                self.batch_size // self.num_caption,
                self.num_caption,
                self.txt_size
            )

            # Transpose({1,2})
            # Torch7 dims start at 1
            x = x.transpose(0, 1)

            # Mean(1)
            x = x.mean(dim=0)

            # Replicate(numCaption)
            x = x.unsqueeze(0).repeat(self.num_caption, 1, 1)

            # Transpose({1,2})
            x = x.transpose(0, 1)

            # Reshape(batchSize, txtSize)
            x = x.reshape(self.batch_size, self.txt_size)

            return x
        else:
            x = x.view(
                self.batch_size // self.num_caption,
                self.num_caption,
                self.txt_size
            )
            x = x.transpose(0, 1)
            x = x.mean(dim=0)
            return x
