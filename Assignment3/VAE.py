# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 00:41:16 2019

@author: Susan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms, utils
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from random import randint

from IPython.display import Image
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io

#%%
'''Build a list of pics(cartoon)'''
f = open('cartoon.txt','w')
for i in range(1,10000):
    img_path="./DL_HW3/cartoon/" + str("%05d" % i) + ".png\n"
    f.write(img_path)
f.close()
#%%

'''Build my own dataset'''
def default_loader(path):
    return Image.open(path).convert('RGB')

class CartoonDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            imgs.append(line)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)

'''Define dataloader'''
transform = transforms.Compose(
                 [transforms.Resize([64,64]), #transforms.Resize([64,64]),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = CartoonDataset(txt='cartoon.txt', transform = transform)
dataloader = DataLoader(dataset, batch_size=64,shuffle=True)
print(len(dataloader))


'''Show images'''
#def show_batch(imgs):
#    grid = utils.make_grid(imgs)
#    plt.imshow(grid.numpy().transpose((1, 2, 0)))
#    plt.title('Batch from dataloader')
#
#for i, (batch_img) in enumerate(dataloader):
#    if(i<4):
#        print(i, batch_img.size())
#        show_batch(batch_img)
#        plt.axis('off')
#        plt.show()
#    else:
#        break

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 256
H_DIM = 512
Z_DIM = 64

#%%
''' Build model '''
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=H_DIM):
        return input.view(input.size(0), size, 1, 1)

class VAE(nn.Module):
    def __init__(self, image_channels = 3, h_dim = H_DIM, z_dim = Z_DIM):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = Variable(torch.randn(*mu.size())).cuda()
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        self.h = self.encoder(x)
        z, mu, logvar = self.bottleneck(self.h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

#%%
image_channels = 3

model = VAE(image_channels=image_channels).to(device)
model.cuda()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def loss_fn(recon_x, x, mu, logvar):
#    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    
    BCE = F.mse_loss(recon_x, x, size_average=False)
    BCE = BCE.cuda()
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD
#%%
epochs = 300
lr_curve = []
for epoch in range(epochs):
    for idx, images in enumerate(dataloader):
        images = Variable(images).cuda()
        recon_images, mu, logvar = model(images)
        loss, bce, kld = loss_fn(recon_images, images, mu, logvar)
        lr_curve.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch+1, epochs, loss.data, bce.data, kld.data)
        print(to_print)


torch.save(model.state_dict(), 'vae_mse_300epoch.torch')
#%%
import numpy as np
plt.plot(lr_curve)
plt.title('Learning curve')
plt.xlabel('iteration')
plt.ylabel('Loss(MSE+KL)')
plt.show()
#loss_ = np.zeros(len(lr_curve))
#for i in range(len(lr_curve)):
#    loss_[i] = lr_curve[i].data.cpu().numpy()
#
#plt.plot(loss_)
#plt.show()
#%%
'''Load saved model'''
#model.load_state_dict(torch.load('vae.torch', map_location='cpu'))
def compare(x):
    x_ = Variable(x).cuda()
    recon_x, _, _ = model(x_)
    recon_x = recon_x.data.cpu()
    return recon_x

recon_pics = []
for i in range(56):
    fixed_x = next(iter(dataloader))
    fixed_x = dataset[randint(1, 10000)].unsqueeze(0)
    recon_pics.append(fixed_x.squeeze(0).data.cpu())
    compare_x  = compare(fixed_x).squeeze(0)
    recon_pics.append(compare_x.data.cpu())
    
#    save_image(compare_x.data.cpu(), 'sample_image.png')
#    Image.open('sample_image.png').show()
    
#%%
grid_img = torchvision.utils.make_grid(recon_pics, nrow=8, padding=2)
plt.figure(figsize = (12,15))
plt.axis('off')
plt.imshow(grid_img.permute(1, 2, 0))

#%%
'''random latent code'''
esp = Variable(torch.randn((64, 512))).cuda()
z, _, _ = model.bottleneck(esp)
sample_x = model.decode(z).squeeze(0).data.cpu()
grid_img = torchvision.utils.make_grid(sample_x, nrow=8, padding=1)
plt.figure(figsize = (10,12))
plt.axis('off')
plt.imshow(grid_img.permute(1, 2, 0))

#%%
def compare(x):
    recon_x, _, _ = model(x)
    return torch.cat([x, recon_x])

fig = plt.figure(figsize=(14, 14))
row = 7
column = 4
axes = fig.subplots(row, column)

fig_cnt = 0

idx = np.random.randint(0, 10000, 28)

for i in range(row):
    for j in range(column):
        fixed_x = (dataset[idx[fig_cnt]]).unsqueeze(0)
        compare_x = compare(fixed_x)
        save_image(compare_x.data.cpu(), str(fig_cnt)+'.png')
        
        axes[i, j].imshow(Image.open(str(fig_cnt)+'.png'), interpolation='nearest')
        axes[i, j].set_axis_off()
        fig_cnt += 1