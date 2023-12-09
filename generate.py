import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data import *
from vae import *
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weight_path = 'params/vae.pth'

net = VAE(in_channels = 3, latent_dim = 128).to(device)
net.load_state_dict(torch.load(weight_path))

img = net.sample(16).cpu()
print(img.size())

toPIL = transforms.ToPILImage()
img_out = img[0]

i = 1
while i < 4:
    img_out = torch.cat((img_out,img[i]),dim=-1)
    i+=1

j = 1
while j < 4:
    tmp = img[j * 4]
    i = 1
    while i < 4:
        tmp = torch.cat((tmp,img[j * 4 + i]),dim=-1)
        i += 1
    img_out = torch.cat((img_out,tmp),dim = -2)
    j += 1


img_out = toPIL(img_out)
img_out.save('test.jpg')