import torch
from torch import nn
import torch.nn.functional as F
from mlassistant.core import ModelIO, Model


class UNet(Model):
    def __init__(self):
        super().__init__()
        
        self.down_sampler1 = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.down_sampler2 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.down_sampler3 = nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )

        self.down_sampler4 = nn.Sequential(
            nn.Conv2d(256, 512, 3),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )

        self.down_sampler5 = nn.Sequential(
            nn.Conv2d(512, 1024, 2),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, 2),
            nn.ReLU(),
            nn.BatchNorm2d(1024)
        )

        self.conv_transpose1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv_transpose2 = nn.ConvTranspose2d(512, 256, 2, stride=2)   
        self.conv_transpose3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_transpose4 = nn.ConvTranspose2d(128, 64, 2, stride=2)      

        self.up_sampler1 = nn.Sequential(
            nn.Conv2d(1024, 512, 3),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )  
        
        self.up_sampler2 = nn.Sequential(
            nn.Conv2d(512, 256, 3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        
        self.up_sampler3 = nn.Sequential(
            nn.Conv2d(256, 128, 3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.up_sampler4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 2, 1, padding=1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, scans_x: torch.Tensor, scans_y: torch.Tensor) -> ModelIO:
        # x:    B   572   572
        scans_x = scans_x.reshape((-1, 1, 572, 572))
        down_sampled1 = self.down_sampler1(scans_x)
        pooled = F.max_pool2d(down_sampled1, 2, 2)
        down_sampled2 = self.down_sampler2(pooled)
        pooled = F.max_pool2d(down_sampled2, 2, 2)
        down_sampled3 = self.down_sampler3(pooled)
        pooled = F.max_pool2d(down_sampled3, 2, 2)
        down_sampled4 = self.down_sampler4(pooled)
        pooled = F.max_pool2d(down_sampled4, 2, 2)
        down_sampled5 = self.down_sampler5(pooled)

        upsampled1 = self.up_sampler1(self.crop(self.conv_transpose1(down_sampled5), 
                                                              down_sampled4))
        upsampled2 = self.up_sampler2(self.crop(self.conv_transpose2(upsampled1),
                                                              down_sampled3))
        upsampled3 = self.up_sampler3(self.crop(self.conv_transpose3(upsampled2),
                                                              down_sampled2))
        out = self.up_sampler4(self.crop(self.conv_transpose4(upsampled3), 
                                                              down_sampled1)).squeeze()
        
        if out.shape[0] == 1:
          out = out.squeeze()
          
        output = {
            'result': out
        }

        if scans_y is not None:
            output['loss'] = F.binary_cross_entropy(out[:, 0, ...], scans_y.squeeze())
        
        return output

    def crop(self, x1, x2):
      height = x2.size()[2] - x1.size()[2]
      width = x2.size()[3] - x1.size()[3]
      x1 = F.pad(x1, [width // 2, width - width // 2,
                      height // 2, height - height // 2])
      return torch.cat([x2, x1], dim=1)
    
