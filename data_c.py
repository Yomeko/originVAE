import os
import torch

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd

CelebA_transforms=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(148),
    transforms.Resize([64,64]),
    transforms.ToTensor(),
])

class CelebA(Dataset):

    def __init__(self,path):
        self.path=path
        self.name=os.listdir(path)
        self.dataframe=pd.read_csv('list_attr_celeba.csv')

    def __len__(self):
        return len(self.name)
    
    def __getitem__(self, index):
        img_path=os.path.join(self.path,self.name[index])
        img=Image.open(img_path)
        cond=self.dataframe.iloc[index]
        return CelebA_transforms(img),torch.Tensor(cond)
    

def split_dataset(dataset: Dataset, train_ratio = 0.7):
    split_seed = torch.Generator().manual_seed(0)
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset=random_split(dataset, [train_size, test_size], generator=split_seed)
    return train_dataset, test_dataset
    
if __name__=='__main__':

    full_dataset = CelebA('img_align_celeba')

    train_dataset, test_dataset = split_dataset(full_dataset)

    train_loader = DataLoader(train_dataset, batch_size = 1, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)

    # data_loader = DataLoader(CelebA('img_align_celeba'), batch_size = 1, shuffle = False)

    for i,img in enumerate(train_loader):

        print(img)

        if i == 0:
            break