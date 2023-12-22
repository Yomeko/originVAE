import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import vgg19_bn
from typing import List, Callable, Union, Any, TypeVar, Tuple

class VAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int) -> None:
        super(VAE, self).__init__()

        self.latent_dim=latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        inC=in_channels
        modules = []
        hidden_dims = [32, 64, 128, 256, 512]

        #encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=inC, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            inC = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        #decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*4)
        
        modules = []
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=hidden_dims[i],
                                       out_channels=hidden_dims[i+1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[-1],
                               out_channels=hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=hidden_dims[-1],
                      out_channels=3,
                      kernel_size=3,
                      padding=1),
            nn.Tanh()
        )

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        mu=self.fc_mu(result)
        log_var=self.fc_var(result)

        return [mu,log_var]
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:

        # print(z.size())
        result = self.decoder_input(z)
        # print(result.size())
        result = result.view(-1,512,2,2)
        result = self.decoder(result)
        result = self.final_layer(result)

        return result
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):

        
        std = torch.exp(0.5 * logvar).to(self.device)
        eps = torch.randn_like(std).to(self.device)
        # print(std.size())

        return eps * std + mu
    
    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:

        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)

        return [output, input, mu, logvar]
    
    def sample(self, num: int) -> torch.Tensor:

        z = torch.randn(num, self.latent_dim).to(self.device)

        samples = self.decode(z)

        return samples
    
    def reconstruct(self, x:torch.Tensor) -> torch.Tensor:
        
        return self.forward(x)[0]
    
class cVAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 img_size: int = 64,
                 num_classes: int = 40) -> None:
        super(cVAE, self).__init__()

        self.latent_dim=latent_dim
        self.img_size=img_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embed_class = nn.Linear(num_classes, img_size * img_size)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size = 1)

        inC=in_channels+1
        modules = []
        hidden_dims = [32, 64, 128, 256, 512]

        #encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=inC, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            inC = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        #decoder
        self.decoder_input = nn.Linear(latent_dim + num_classes, hidden_dims[-1]*4)
        
        modules = []
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=hidden_dims[i],
                                       out_channels=hidden_dims[i+1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dims[-1],
                               out_channels=hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=hidden_dims[-1],
                      out_channels=3,
                      kernel_size=3,
                      padding=1),
            nn.Tanh()
        )

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        mu=self.fc_mu(result)
        log_var=self.fc_var(result)

        return [mu,log_var]
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:

        # print(z.size())
        result = self.decoder_input(z)
        # print(result.size())
        result = result.view(-1,512,2,2)
        result = self.decoder(result)
        result = self.final_layer(result)

        return result
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):

        
        std = torch.exp(0.5 * logvar).to(self.device)
        eps = torch.randn_like(std).to(self.device)
        # print(std.size())

        return eps * std + mu
    
    def forward(self, input: List[torch.Tensor]) -> List[torch.Tensor]:

        embedded_class = self.embed_class(input[1])
        embedded_class = embedded_class.view(-1, self.img_size, self.img_size).unsqueeze(1)
        embedded_img = self.embed_data(input[0])

        embedded_input = torch.cat([embedded_img, embedded_class], dim = 1)

        mu, logvar = self.encode(embedded_input)
        z = self.reparameterize(mu, logvar)
        z = torch.cat([z, input[1]], dim = 1)
        output = self.decode(z)

        return [output, input[0], mu, logvar]
    
    def sample(self, num: int, y: torch.Tensor) -> torch.Tensor:

        z = torch.randn(num, self.latent_dim).to(self.device)
        z = torch.cat([z, y], dim = 1)

        samples = self.decode(z)

        return samples
    
    def reconstruct(self, x:List[torch.Tensor]) -> torch.Tensor:
        
        return self.forward(x)[0]
    

class VAELoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args, kld_weight=0.00025):

        recons = args[0]
        input = args[1]
        mu = args[2]
        logvar = args[3]
        kld_weight = 0.00025

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss

        return loss
    

class DFCVAELoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.feature_network = vgg19_bn().cuda()
        self.feature_network.load_state_dict(torch.load('params/vgg19_bn-c79401a0.pth'))

        for param in self.feature_network.parameters():
            param.requires_grad = False

        self.feature_network.eval()

    def feature_extract(self, input: torch.Tensor) -> List[torch.Tensor]:
        feature_layers = ['14', '24', '34', '43']
        features = []

        result = input

        for (key, module) in self.feature_network.features._modules.items():
            result = module(result)
            if(key in feature_layers):
                features.append(result)

        return features
    
    def forward(self, *args, kld_weight = 0.00025):
        recons = args[0]
        input = args[1]
        mu = args[2]
        logvar = args[3]

        recons_features = self.feature_extract(recons)
        input_features = self.feature_extract(input)

        recons_loss = F.mse_loss(recons, input)

        features_loss = 0.0
        for (r, i) in zip(recons_features,input_features):
            features_loss += F.mse_loss(r, i)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)

        loss = 0.5*(recons_loss + features_loss) + kld_weight * kld_loss

        return loss


if __name__ == '__main__':
    net = VAE(3,128)
    net = net.to(net.device)

    test = torch.randn([4,3,64,64]).to(net.device)
    
    out = net(test)
    print(len(out))