import numpy as np
import torch
import os
import torchvision.utils as vutils
import matplotlib.pyplot as plt
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
def smooth_label(tensor, offset):
        return tensor + offset
#plt.ion()
def save_checkpoint(netD, netG, dir_path, subdir_path, epoch):
        path =  os.path.join(dir_path, subdir_path)
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(netD.state_dict(), '{0}/disc_{1}.pth'.format(path, epoch))
        torch.save(netG.state_dict(), '{0}/gen_{1}.pth'.format(path, epoch))

class Logger(object):
    def __init__(self):
        self.hist_D = []
        self.hist_G = []
        self.hist_Dx = []
        self.hist_DGx = []


    def log_iteration_gan(self, epoch, d_loss, g_loss, real_score, fake_score):
        print("Epoch: %d, d_loss= %f, g_loss= %f, D(X)= %f, D(G(X))= %f" % (
            epoch, d_loss.data.cpu().mean(), g_loss.data.cpu().mean(), real_score.data.cpu().mean(),
            fake_score.data.cpu().mean()))
        self.hist_D.append(d_loss.data.cpu().mean())
        self.hist_G.append(g_loss.data.cpu().mean())
        self.hist_Dx.append(real_score.data.cpu().mean())
        self.hist_DGx.append(fake_score.data.cpu().mean())

    def plot_epoch(self, epoch):
        plt.plot('Discriminator', 'train', epoch, np.array(self.hist_D).mean())
        plt.plot('Generator', 'train', epoch, np.array(self.hist_G).mean())
        self.hist_D = []
        self.hist_G = []

    def plot_epoch_w_scores(self, epoch):
        # plt.plot('Discriminator', 'train', epoch, np.array(self.hist_D).mean())
        # plt.plot('Generator', 'train', epoch, np.array(self.hist_G).mean())
        # plt.plot('D(X)', 'train', epoch, np.array(self.hist_Dx).mean())
        # plt.plot('D(G(X))', 'train', epoch, np.array(self.hist_DGx).mean())
        plt.plot(self.hist_D)
        plt.plot(self.hist_G)
        plt.plot(self.hist_Dx)
        plt.plot(self.hist_DGx)
        plt.legend(['D_loss', 'G_loss', 'D(X)', 'D(G(X))'])
        plt.xlabel('Iterations')
        plt.title(f'Epoch {epoch}')
        plt.show()
        self.hist_D = []
        self.hist_G = []
        self.hist_Dx = []
        self.hist_DGx = []

    def draw(self, right_images, fake_images):
        right = right_images[0].detach().cpu()
        fake = fake_images[0].detach().cpu()

        real_grid = vutils.make_grid(
            right,
            nrow=8,
            normalize=True,
            padding=2
        )
        fake_grid = vutils.make_grid(
             fake,
             nrow=8,
             normalize=True,
             padding=2
        )


        # [C,H,W] -> [H,W,C]
        real_grid = real_grid.permute(1, 2, 0)
        fake_grid = fake_grid.permute(1, 2, 0)

        fig, axes = plt.subplots(1, 2, figsize=(16,8))

        axes[0].imshow(real_grid)
        axes[0].set_title("Real Images")
        axes[0].axis("off")

        axes[1].imshow(fake_grid)
        axes[1].set_title("Generated Images")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()