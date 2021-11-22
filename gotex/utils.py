# imports
import torch
from torch import nn
import torch.nn.functional as nnFunc
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os

# global variables
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TENSORTYPE = torch.float


def ReadImg(imagePath):
    npImg = plt.imread(imagePath)
    if np.max(npImg) <= 1:
        npImg*=255
    tensImg = torch.tensor(npImg, dtype=TENSORTYPE)
    if len(tensImg.shape) < 3:
        tensImg = tensImg.unsqueeze(2)
        tensImg = torch.cat((tensImg, tensImg, tensImg), 2)
    if tensImg.shape[2] > 3:
        tensImg = tensImg[:,:,:3]
    return tensImg#[:,:,[2,1,0]]

def ShowImg(tensImg):
    npImg = np.clip((tensImg.data.cpu().numpy())/255, 0,1) #[:,:,[2,1,0]]
    ax = plt.imshow(npImg)
    return ax

def SaveImg(saveName, tensImg):
    npImg = np.clip((tensImg.cpu().numpy())/255, 0,1)
    if npImg.shape[2] < 3:
        npImg = npImg[:,:,0]
    plt.imsave(saveName, npImg)
    return 

def PrepImg(tensImg):
    out = tensImg[:,:,[2,1,0]]
    out = out - torch.tensor([255*0.40760392, 255*0.45795686,255*0.48501961], device=tensImg.device).view(1,1,3)
    return out.permute(2,0,1).unsqueeze(0)

def PostImg(batchImg):
    out = batchImg.squeeze(0).permute(1,2,0)
    out = out + torch.tensor([255*0.40760392, 255*0.45795686,255*0.48501961], device=batchImg.device).view(1,1,3)
    return out[:,:,[2,1,0]]

class TorusPadding(nn.Module):
    def __init__(self, padSize):
        super(TorusPadding, self).__init__()
        self.padSize = padSize
    def forward(self, input):
        input = torch.cat((input, input[:,:,:self.padSize,:]), 2)
        input = torch.cat((input, input[:,:,:,:self.padSize]), 3)           
        return input
'''
class CustomVGG(nn.Module):
    def __init__(self, pool='max', padding=False):
        super(CustomVGG, self).__init__()
        self.normfeat = [360, 1566, 2739, 8088, 366]
        #vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
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
            
    def forward(self, x):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))

        features = []
        for i,key in enumerate(['r11','r21','r31','r41', 'r51']):
            A = out[key]
            # print(A.shape)
            _,c,h,w = out[key].size()
            if (h*w > 64*64):
                VGG_stride = int(np.ceil(np.sqrt(h*w-1)/64))
                offset = np.random.randint(low=0, high=VGG_stride, size=2)
                A = out[key][:,:,::VGG_stride,::VGG_stride]#[:,:,offset[0]::VGG_stride,offset[1]::VGG_stride]
            features.append(A.squeeze(0).reshape(c,-1).transpose(0,1)/self.normfeat[i])

        # for _,feat in enumerate(features) :
        #     print('VGG features‘ dimension ',feat.shape)

        return features
'''

class CustomVGG(nn.Module):
    def __init__(self, pool='max', padding=False):
        super(CustomVGG, self).__init__()
        self.padding = padding
        if padding:
            self.pad = TorusPadding(2)
        else:
            self.pad = nn.Identity()

        # self.pad = nn.Identity()

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

        # self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        # self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        # self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

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

            ## [tensor(360.4581, device='cuda:0'), tensor(1566.4669, device='cuda:0'), tensor(2739.8643, device='cuda:0'), tensor(8088.1299, device='cuda:0'), tensor(366.4062, device='cuda:0')
        # self.normfeat = [500, 2100, 2700,2800,170]
        self.normfeat = [360, 1566, 2739, 8088, 366]


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
            if (h*w > 64*64) and self.padding:
                VGG_stride = int(np.ceil(np.sqrt(h*w-1)/64))
                offset = np.random.randint(low=0, high=VGG_stride, size=2)
                A = A[:,:,offset[0]::VGG_stride,offset[1]::VGG_stride]
            features.append(A.squeeze(0).reshape(c,-1).transpose(0,1)/self.normfeat[i])
            # print(features[i].shape)

        return features



class SemiDualOptimalTransportLayer(nn.Module):
    """
    forward(dualVariablePsi, inputDataX, targetDataY, batchSplitSize=None)
    """
    def __init__(self, targetDataY):
        super(SemiDualOptimalTransportLayer, self).__init__()
        self.targetDataY = targetDataY
        self.numTargetDataY = targetDataY.size(0)
        self.dualVariablePsi = nn.Parameter(torch.zeros(self.numTargetDataY))

    def forward(self, inputDataX, batchSplitSize=None):
        loss = 0
        numInputDataX = inputDataX.size(0)
        dimInputDataX = inputDataX.size(1)
        if batchSplitSize is None:
            batchSplitSize = numInputDataX   
        InputDataX = torch.split(inputDataX, batchSplitSize)
        for x in InputDataX:            
            costMatrix = (torch.sum(x**2,1,keepdim=True) + torch.sum(self.targetDataY.transpose(0,1)**2,0,keepdim=True) - 2*torch.matmul(x,self.targetDataY.transpose(0,1)))/2 #2#/5000#(dimInputDataX*255) #dimInputDataX*
            loss += torch.sum(torch.min(costMatrix - self.dualVariablePsi.unsqueeze(0),1)[0])
        return loss/numInputDataX + torch.mean(self.dualVariablePsi)

def DownloadVggWeights(modelFolder):
    if not os.path.isdir(modelFolder):
        os.mkdir(modelFolder)
    if not os.path.isfile('./'+modelFolder+'/vgg_conv.pth'):
        defaultPath = os.getcwd()
        os.chdir('./'+modelFolder)
        !wget -c --no-check-certificate https://bethgelab.org/media/uploads/pytorch_models/vgg_conv.pth
        os.chdir(defaultPath)
    return

def GetDemoTexture():
    if not os.path.isfile('./demo_texture_2.png'):
        !wget https://raw.githubusercontent.com/ahoudard/wgenpatex/main/texture_images/demo_texture_2.png
    else:
        print('Image already here!')
    return

def CreateVggNet(modelFolder, padding=True):
    DownloadVggWeights(modelFolder)
    vggNet = CustomVGG(pool='avg', padding=padding)
    vggNet.load_state_dict(torch.load('./'+modelFolder+'/vgg_conv.pth'))
    vggNet.eval()
    for param in vggNet.parameters():
        param.requires_grad = False
    return vggNet.to(DEVICE)

def CreateFeaturesExtractor(featureType, padding=True):
    if featureType == "vgg":
        FeatureExtractor = CreateVggNet("VggModel", padding=padding)
    else:
        FeatureExtractor = PatchPyramid(4, 4, 1, 2, pad, 3, 1, pad, 1)# todo better: numScales, gaussKernelSize, gaussStd, gaussStride, gaussPad, patchSize, patchStride, patchPad, patchDilation):
    return FeatureExtractor

def InitializeOtLayers(inputImg, featureType, psiLearningRate):
    FeatExtr = CreateFeaturesExtractor(featureType, padding=False)
    inputFeatures = FeatExtr(inputImg)
    optimizers = []
    layers  = []
    for feat in inputFeatures:
        currentLayer = SemiDualOptimalTransportLayer(feat).to(DEVICE)
        layers.append(currentLayer)
        optimizers.append(torch.optim.SGD(currentLayer.parameters(), lr=psiLearningRate))
        # optimizers.append(torch.optim.ASGD(currentLayer.parameters(), lr=0.8, alpha=0.5))
    return layers, optimizers

def InitializeImage(numRow, numCol, targetImg = None):
    initImage = torch.rand(numRow,numCol, 3, device=DEVICE)
    initImage.data*=255
    # initImage.data+=128
    if targetImg is not None:
        nRowTarget = targetImg.size(0)
        nColTarget = targetImg.size(1)
        initImage.data[1:nRowTarget+1,1:nColTarget+1, :] = targetImg.data
    return initImage

def InitializeImageOptimizer(initImage, learningRate):
    return torch.optim.LBFGS([initImage], lr=learningRate)
