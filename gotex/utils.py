import os
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import gotex.vgg as vgg
import wget
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def ReadImg(imagePath):
    '''
    Read an image as tensor and ensure that it has 3 channel and range from 0 to 255
    output tensImg has dimension [nrow, ncol, nchannel]
    '''
    npImg = plt.imread(imagePath)
    tensImg = torch.tensor(npImg)
    if torch.max(tensImg) <= 1:
        tensImg*=255    
    if len(tensImg.shape) < 3:
        tensImg = tensImg.unsqueeze(2)
        tensImg = torch.cat((tensImg, tensImg, tensImg), 2)
    if tensImg.shape[2] > 3:
        tensImg = tensImg[:,:,:3]
    return tensImg

def ShowImg(tensImg):
    '''
    Show a tensor image
    tensImg dimension should be [nrow, ncol, nchannel]
    '''
    npImg = np.clip((tensImg.data.cpu().numpy())/255, 0,1)
    ax = plt.imshow(npImg)
    return ax

def SaveImg(saveName, tensImg):
    '''
    Show a tensor image as saveName
    tensImg dimension should be [nrow, ncol, nchannel]
    '''
    npImg = np.clip((tensImg.cpu().numpy())/255, 0,1)
    if npImg.shape[2] < 3:
        npImg = npImg[:,:,0]
    plt.imsave(saveName, npImg)
    return 

def PreProc(tensImg):
    '''
    pre-process an image in order to feed it in VGG net
    input: tensImg as dimension [nrow, ncol, nchannel] with channel RGB
    output: normalized preproc image of dimension [1, nchannel, nrow, ncol] with channel BGR
    '''
    out = tensImg[:,:,[2,1,0]] # RGB to BRG
    out = out - torch.tensor([104, 117, 124], device=tensImg.device).view(1,1,3) # substract VGG mean
    return out.permute(2,0,1).unsqueeze(0) # permute and unsqueeze

def PostProc(batchImg):
    '''
    post-process an image in order to display and save it
    input: batchImg as dimension [1, nchannel, nrow, ncol] with channel BGR
    output: post-processed image of dimension [1, nchannel, nrow, ncol] with channel BGR
    '''
    out = batchImg.squeeze(0).permute(1,2,0) # permute and squeeze
    out = out + torch.tensor([104, 117, 124], device=batchImg.device).view(1,1,3) # add VGG mean
    return out[:,:,[2,1,0]] #BRG to RGB    
  
def DownloadVggWeights(modelFolder):
    if not os.path.isdir(modelFolder):
        os.mkdir(modelFolder)
    if not os.path.isfile('./'+modelFolder+'/vgg_conv.pth'):
        defaultPath = os.getcwd()
        os.chdir('./'+modelFolder)
        wget.download('https://bethgelab.org/media/uploads/pytorch_models/vgg_conv.pth')
        os.chdir(defaultPath)
    return

def CreateVggNet(modelFolder, padding=True):
    DownloadVggWeights(modelFolder)
    vggNet = vgg.CustomVGG(pool='avg', padding=padding)
    vggNet.load_state_dict(torch.load('./'+modelFolder+'/vgg_conv.pth'))
    vggNet.eval()
    for param in vggNet.parameters():
        param.requires_grad = False
    return vggNet
