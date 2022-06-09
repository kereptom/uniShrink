import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms as T
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import scipy.io
import math
from PIL import Image
from os.path import join
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import urllib.request

pl.seed_everything(hash("setting random seeds") % 2**32 - 1)


hyperparameter_defaults = dict(
    max_epochs = 1,
    lr = 0.001,                                             
    momentum = 0.9,                                              
    log_freq = 1,
    l = 1/10,
    b = 0,
    p = 1,
    a1 = 1,
    a2 = 0,
    c = 0,
    s = 0,
    ch = 1,
    H = np.array([[0,0,0,0,0],
                  [0,0,0,0,0], 
                  [0,0,1,0,0],
                  [0,0,0,0,0],
                  [0,0,0,0,0]])
    )

wandb.init(config=hyperparameter_defaults,project="UniShrink")
config = wandb.config

# ____ _   _ ____ ___ ____ _  _ 
# [__   \_/  [__   |  |___ |\/| 
# ___]   |   ___]  |  |___ |  |

class UniShrinkLayer(nn.Module):
  def __init__(self,lamb,H,b,p,a1,a2,c,s,channels):
    super().__init__()
    self.H = torch.tensor(H)
    self.h_conv = nn.Conv2d(channels, channels, self.H.shape, padding=(self.H.shape[0]//2,self.H.shape[1]//2), padding_mode='reflect', groups=channels)
    for a in range(channels):
        self.h_conv.weight.data[a,0,:] = self.H
    self.h_conv.bias.data.fill_(b)
    self.lamb = nn.parameter.Parameter(torch.Tensor([lamb]))
    self.b = nn.parameter.Parameter(torch.Tensor([b]))
    self.p = nn.parameter.Parameter(torch.Tensor([p]))
    self.a1 = nn.parameter.Parameter(torch.Tensor([a1]))
    self.a2 = nn.parameter.Parameter(torch.Tensor([a2]))
    self.c = nn.parameter.Parameter(torch.Tensor([c]))
    self.s = nn.parameter.Parameter(torch.Tensor([s]))

  def forward(self,x):
    x = x-self.s
    denom = self.h_conv(x.pow(2)).sqrt()
    m1 = torch.nan_to_num(1-(self.lamb/denom).pow(self.p), nan=0)
    m1i = torch.nan_to_num(1/m1, nan=0)
    m2 = torch.zeros_like(x)
    maxim = torch.maximum(m1, m2)
    return (x * maxim) * (self.a1 + self.a2 * m1i) + self.c


class SoftLearn(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.softhL1 = UniShrinkLayer(config.l,config.H,config.b,config.p,config.a1,config.a2,config.c,config.s,config.ch)
        self.train_psnr = torchmetrics.PeakSignalNoiseRatio()
        self.LOSS_MSE = nn.MSELoss()
        self.lr = config.lr
        self.momentum = config.momentum

    def forward(self, g):
        s = self.softhL1(g)
        return s
    
    def loss(self, xs, ys):
        outputs = self(xs) 
        loss = self.LOSS_MSE(outputs, ys)
        return outputs, loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)

    def training_step(self, batch, batch_idx):
        xs, ys = batch[0],batch[1]
        outputs, loss = self.loss(xs, ys)
        self.log('train_loss', loss, on_epoch=False)        
        return {'loss': loss, 'outputs': outputs, 'targets': ys}

    def training_step_end(self, outputs):
        self.train_psnr(outputs['outputs'], outputs['targets'])
        self.log('train_psnr', self.train_psnr, on_epoch=False)
    

# ___  ____ ___ ____ 
# |  \ |__|  |  |__| 
# |__/ |  |  |  |  |

class BasicDataset(Dataset):
    def __init__(self,img_in,img_out):
        self.img_in = img_in
        self.img_out = img_out

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.img_in, self.img_out

class UniShrinkDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.transform = transforms.Compose([transforms.ToTensor()])
      
    def setup(self, stage=None):
        urllib.request.urlretrieve("https://i.ytimg.com/vi/j25QBICdKJM/mqdefault.jpg", "cat")
        img = ImageOps.grayscale(Image.open("cat"))
        img_in = self.transform(np.array(img))
        img_out = F.softshrink(img_in,lambd=100)
        self.data_train = BasicDataset(img_in,img_out)

    def train_dataloader(self):
        train_loader = DataLoader(self.data_train)
        return train_loader


# ___ ____ ____ _ _  _ 
#  |  |__/ |__| | |\ | 
#  |  |  \ |  | | | \|

print(f'\nStarting- a run with hyperparameters:')
for key, value in config.items():
    print('\t', key, ' : ', value)

# setup data
UniData = UniShrinkDataModule(config)
UniData.setup()

# setup model
model = SoftLearn(config)

# setup wandb
wandb_logger = WandbLogger(project="UniShrink")
wandb_logger.watch(model, log_freq=config.log_freq)

# fit the model
trainer = pl.Trainer(logger=wandb_logger, log_every_n_steps=config.log_freq, accelerator="gpu", devices=-1, auto_select_gpus=True, max_epochs=config.max_epochs)
trainer.fit(model, UniData)

