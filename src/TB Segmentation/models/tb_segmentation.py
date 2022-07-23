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
            nn.Conv2d(64, 2, 3),
            nn.ReLU()
        )
    
    def forward(self, mnist_x: torch.Tensor, mnist_y: torch.Tensor) -> ModelIO:
        # x:    B   572   572
        out = self._seq(mnist_x)

        output = {
            'categorical_probability': out,
        }

        if mnist_y is not None:
            output['loss'] = F.cross_entropy(out, mnist_y.long())
        
        return output
