import torch
from torch import nn
import torch.nn.functional as nnFunc
import numpy as np

class TorusPadding(nn.Module):
    '''
    Padding Module thats pad the images as it is on a torus
    '''
    def __init__(self, padSize):
        super(TorusPadding, self).__init__()
        self.padSize = padSize
    def forward(self, input):
        input = torch.cat((input, input[:,:,:self.padSize,:]), 2)
        input = torch.cat((input, input[:,:,:,:self.padSize]), 3)           
        return input

class CustomVGG(nn.Module):
    '''
    VGG module with custom features extractions
    '''
    def __init__(self, pool='max', padding=False):
        super(CustomVGG, self).__init__()
        self.padding = padding
        if padding:
            self.pad = TorusPadding(2)
        else:
            self.pad = nn.Identity()

        self.outKeys = ['r11','r21','r31','r41', 'r51']
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=0)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=0)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=0)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=0)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=0)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=0)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=0)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=0)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=0)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=0)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=0)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=0)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=0)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=0)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=0)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=0)

        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.normfeat = [1, 1, 1,1, 1]

    def forward(self, x):
        out = {}
        out['r11'] = nnFunc.relu(self.conv1_1(self.pad(x)))
        out['r12'] = nnFunc.relu(self.conv1_2(self.pad(out['r11'])))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = nnFunc.relu(self.conv2_1(self.pad(out['p1'])))
        out['r22'] = nnFunc.relu(self.conv2_2(self.pad(out['r21'])))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = nnFunc.relu(self.conv3_1(self.pad(out['p2'])))
        out['r32'] = nnFunc.relu(self.conv3_2(self.pad(out['r31'])))
        out['r33'] = nnFunc.relu(self.conv3_3(self.pad(out['r32'])))
        out['r34'] = nnFunc.relu(self.conv3_4(self.pad(out['r33'])))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = nnFunc.relu(self.conv4_1(self.pad(out['p3'])))
        out['r42'] = nnFunc.relu(self.conv4_2(self.pad(out['r41'])))
        out['r43'] = nnFunc.relu(self.conv4_3(self.pad(out['r42'])))
        out['r44'] = nnFunc.relu(self.conv4_4(self.pad(out['r43'])))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = nnFunc.relu(self.conv5_1(self.pad(out['p4'])))
        
        features = []
        for i,key in enumerate(self.outKeys):
            A = out[key]
            _,c,h,w = A.size()            
            if (h*w > 64*64) and (self.padding):
                VGG_stride = int(np.ceil(np.sqrt(h*w-1)/64))
                offset = np.random.randint(low=0, high=VGG_stride, size=2)
                A = A[:,:,offset[0]::VGG_stride,offset[1]::VGG_stride]
            features.append(A.squeeze(0).reshape(c,-1).transpose(0,1)/self.normfeat[i])

        return features
