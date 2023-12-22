import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data import *
from vae import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weight_path = 'params/dfcvae.pth'
data_path = '/root/autodl-tmp/celeba/img_align_celeba'

full_dataset = CelebA(data_path)
train_dataset, test_dataset = split_dataset(full_dataset, 0.8)

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
net = VAE(in_channels = 3, latent_dim = 128).to(device)
if os.path.exists(weight_path):
    net.load_state_dict(torch.load(weight_path))
    print('weight loaded')

else:
    print('not loaded')

opt = optim.Adam(net.parameters(),lr = 0.005)
scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma = 0.95)
loss_fun = DFCVAELoss()

epoch = 1
max_epoch = 10

while True:
    for i,img in enumerate(train_loader):
        img = img.to(device)

        out = net(img)
        train_loss = loss_fun(*out)

        opt.zero_grad()
        train_loss.backward()
        opt.step()

        if (i + 1) % 250 == 0:
            print(f'{epoch}-{i+1}-train_loss ===>> {train_loss.item()}')

    torch.save(net.state_dict(), weight_path)
    scheduler.step()
    if epoch == max_epoch:
        break
    
    epoch += 1