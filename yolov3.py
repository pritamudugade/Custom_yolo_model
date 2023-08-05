#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x

class YOLOv3(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv3, self).__init__()

        # Backbone
        self.backbone = nn.Sequential(
            ConvBlock(3, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128, 3, 1, 1),
            ConvBlock(128, 64, 1, 1, 0),
            ConvBlock(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256, 3, 1, 1),
            ConvBlock(256, 128, 1, 1, 0),
            ConvBlock(128, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
        )

        # Detection layers
        self.detector_1 = nn.Sequential(
            ConvBlock(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, num_classes, 1, 1, 0)
        )
        self.detector_2 = nn.Sequential(
            ConvBlock(512, 256, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.detector_3 = nn.Sequential(
            ConvBlock(768, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            nn.Conv2d(512, num_classes, 1, 1, 0)
        )

    def forward(self, x):
        # Backbone
        x_2 = self.backbone[:10](x)
        x_3 = self.backbone[10:](x_2)

        # Detection heads
        output_1 = self.detector_1(x_3)

        x_3_upsampled = self.detector_2(x_3)
        x_2_concat = torch.cat((x_3_upsampled, x_2), dim=1)
        output_2 = self.detector_3(x_2_concat)

        return output_1, output_2

# Example usage
num_classes = 1  
model = YOLOv3(num_classes)






