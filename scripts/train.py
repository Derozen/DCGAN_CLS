from logging import Logger
from models import dcgan_cls
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from .datasetconfig import LoadDataset
from .utils import *

class Trainer(object):
    def __init__(self, type, dataset, split, lr, diter, save_path, l1_coef, l2_coef, pre_trained_gen, pre_trained_disc, batch_size, num_workers, epochs):

        self.generator = torch.nn.DataParallel(dcgan_cls.Generator().cuda())
        self.discriminator = torch.nn.DataParallel(dcgan_cls.Discriminator().cuda())

        if pre_trained_disc:
            self.discriminator.load_state_dict(torch.load(pre_trained_disc))
        else:
            self.discriminator.apply(init_weights)


        if pre_trained_gen:
            self.generator.load_state_dict(torch.load(pre_trained_gen))
        else:
            self.generator.apply(init_weights)

        self.dataset = LoadDataset(dataset['dataset_path'], split=split)
        self.logger = Logger()
        self.noise_dim = 100
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.beta1 = 0.5
        self.num_epochs = epochs
        self.DITER = diter

        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                num_workers=0)

        self.optimD = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimG = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        print("----------------------------------------------------------------------")
        print("----------------------Training Start----------------------")
        self.checkpoints_path = 'checkpoints'
        self.save_path = save_path
        self.type = type


    def train(self, cls=1):
        criterion = nn.BCELoss()
        l2_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()
        iteration = 0

        for epoch in range(self.num_epochs):
            for sample in self.data_loader:
                iteration += 1
                right_images = sample['right_images']
                right_embed = sample['right_embed']
                wrong_images = sample['wrong_images']

                right_images = Variable(right_images.float()).cuda()
                right_embed = Variable(right_embed.float()).cuda()
                wrong_images = Variable(wrong_images.float()).cuda()

                real_labels = torch.ones(right_images.size(0))
                fake_labels = torch.zeros(right_images.size(0))


                smoothed_real_labels = torch.FloatTensor(smooth_label(real_labels.numpy(), -0.1))

                real_labels = Variable(real_labels).cuda()
                smoothed_real_labels = Variable(smoothed_real_labels).cuda()
                fake_labels = Variable(fake_labels).cuda()

                # Train the discriminator
                self.discriminator.zero_grad()
                outputs, activation_real = self.discriminator(right_images, right_embed)
                smoothed_real_labels = smoothed_real_labels.unsqueeze(1).float() 
                real_loss = criterion(outputs, smoothed_real_labels)
                real_score = outputs
                outputs, _ = self.discriminator(wrong_images, right_embed)
                fake_labels = fake_labels.unsqueeze(1).float()
                wrong_loss = criterion(outputs, fake_labels)
                wrong_score = outputs

                noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_embed, noise)
                # print(fake_images.shape)
                # print(right_embed.shape)
                outputs, _ = self.discriminator(fake_images, right_embed)
                fake_loss = criterion(outputs, fake_labels)
                fake_score = outputs

                d_loss = real_loss + fake_loss

                d_loss = d_loss + wrong_loss

                d_loss.backward()
                self.optimD.step()

                # Train the generator
                self.generator.zero_grad()
                noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_embed, noise)
                outputs, activation_fake = self.discriminator(fake_images, right_embed)
                _, activation_real = self.discriminator(right_images, right_embed)

                activation_fake = torch.mean(activation_fake, 0)
                activation_real = torch.mean(activation_real, 0)


                
                real_labels = real_labels.unsqueeze(1).float()
                g_loss = criterion(outputs, real_labels) \
                         + self.l2_coef * l2_loss(activation_fake, activation_real.detach()) \
                         + self.l1_coef * l1_loss(fake_images, right_images)

                g_loss.backward()
                self.optimG.step()
                if iteration % 5 == 0:
                    # print(fake_images.shape)
                    # print(right_images.shape)
                    self.logger.log_iteration_gan(epoch,d_loss, g_loss, real_score, fake_score)
                    self.logger.draw(right_images, fake_images)

            self.logger.plot_epoch_w_scores(epoch)

            if (epoch) % 10 == 0:
                save_checkpoint(self.discriminator, self.generator, self.checkpoints_path, self.save_path, epoch)

  
    def predict(self):
        for sample in self.data_loader:
            right_images = sample['right_images']
            right_embed = sample['right_embed']
            txt = sample['txt']

            if not os.path.exists('results/{0}'.format(self.save_path)):
                os.makedirs('results/{0}'.format(self.save_path))

            right_images = Variable(right_images.float()).cuda()
            right_embed = Variable(right_embed.float()).cuda()

            # Train the generator
            noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
            noise = noise.view(noise.size(0), 100, 1, 1)
            fake_images = self.generator(right_embed, noise)

            self.logger.draw(right_images, fake_images)

            for image, t in zip(fake_images, txt):
                im = Image.fromarray(image.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
                im.save('results/{0}/{1}.jpg'.format(self.save_path, t.replace("/", "")[:100]))
                print(t)