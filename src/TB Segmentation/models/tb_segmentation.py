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
            nn.Conv2d(64, 64, 3),
            nn.ReLU()
        )

        self.down_sampler2 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU()
        )

        self.down_sampler3 = nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3),
            nn.ReLU()
        )

        self.down_sampler4 = nn.Sequential(
            nn.Conv2d(256, 512, 3),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3),
            nn.ReLU()
        )

        self.down_sampler5 = nn.Sequential(
            nn.Conv2d(512, 1024, 2),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 2),
            nn.ReLU()
        )

        self.conv_transpose1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv_transpose2 = nn.ConvTranspose2d(512, 256, 2, stride=2)   
        self.conv_transpose3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_transpose4 = nn.ConvTranspose2d(128, 64, 2, stride=2)      

        self.up_sampler1 = nn.Sequential(
            nn.Conv2d(1024, 512, 3),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3),
            nn.ReLU()
        )  
        
        self.up_sampler2 = nn.Sequential(
            nn.Conv2d(512, 256, 3),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3),
            nn.ReLU()
        )
        
        self.up_sampler3 = nn.Sequential(
            nn.Conv2d(256, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3),
            nn.ReLU()
        )

        self.up_sampler4 = nn.Sequential(
            nn.Conv2d(128, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 2, 1)
        )
    
    def forward(self, scan_x: torch.Tensor, scan_y: torch.Tensor) -> ModelIO:
        # x:    B   572   572
        down_sampled1 = self.down_sampler1(scan_x)
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
                                                              down_sampled1))

        output = {
            'categorical_probability': out,
        }

        if scan_y is not None:
            output['loss'] = F.cross_entropy(out, scan_y)
        
        return output

    def crop(self, x1, x2):
      height = x2.size()[2] - x1.size()[2]
      width = x2.size()[3] - x1.size()[3]
      x1 = F.pad(x1, [width // 2, width - width // 2,
                      height // 2, height - height // 2])
      return torch.cat([x2, x1], dim=1)
    
