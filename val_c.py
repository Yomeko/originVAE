import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data_c import *
from vae import *
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
toPIL = transforms.ToPILImage()

weight_path = 'params/cvae.pth'
data_path = '/root/autodl-tmp/celeba/img_align_celeba'

full_dataset = CelebA(data_path)
train_dataset, test_dataset = split_dataset(full_dataset, 0.8)

test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)
net = cVAE(in_channels = 3, latent_dim = 128).to(device)
if os.path.exists(weight_path):
    net.load_state_dict(torch.load(weight_path))
    print('weight loaded')

else:
    print('not loaded')

loss_fun = VAELoss()

net.eval()
total_loss = 0

with torch.no_grad():
    for i,img in enumerate(test_loader):
        img[0] = img[0].to(device)
        img[1] = img[1].to(device)

        out = net(img)
        eval_loss = loss_fun(*out)

        total_loss += eval_loss

        if i == 0:
            out = net.reconstruct(img)

            img_out = out[0]
            img_ori = img[0][0]
            j = 1

            while j < 8:
                img_out = torch.cat((img_out,out[j]), dim = -1)
                img_ori = torch.cat((img_ori,img[0][j]), dim = -1)
                j += 1

            img_out = torch.cat((img_ori,img_out), dim = -2)

            k = 1
            while k < 4:
                tmp = out[k * 8]
                tmp2 = img[0][k * 8]
                j = 1
                while j < 8:
                    tmp = torch.cat((tmp,out[k * 8 + j]), dim = -1)
                    tmp2 = torch.cat((tmp2,img[0][k * 8 + j]), dim = -1)
                    j += 1

                img_out = torch.cat((img_out,tmp2), dim = -2)
                img_out = torch.cat((img_out,tmp), dim = -2)
                k += 1

            img_out = toPIL(img_out)
            img_out.save('recons_c.jpg')


            out = net.sample(32, img[1])

            img_out = out[0]
            img_ori = img[0][0]
            j = 1

            while j < 8:
                img_out = torch.cat((img_out,out[j]), dim = -1)
                img_ori = torch.cat((img_ori,img[0][j]), dim = -1)
                j += 1

            img_out = torch.cat((img_ori,img_out), dim = -2)

            k = 1
            while k < 4:
                tmp = out[k * 8]
                tmp2 = img[0][k * 8]
                j = 1
                while j < 8:
                    tmp = torch.cat((tmp,out[k * 8 + j]), dim = -1)
                    tmp2 = torch.cat((tmp2,img[0][k * 8 + j]), dim = -1)
                    j += 1

                img_out = torch.cat((img_out,tmp2), dim = -2)
                img_out = torch.cat((img_out,tmp), dim = -2)
                k += 1

            img_out = toPIL(img_out)
            img_out.save('genes_c.jpg')


total_loss /= len(test_loader) * 32
print(f'mean loss on test dataset is {total_loss}')