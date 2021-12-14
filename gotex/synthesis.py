import torch
from torch import nn
import torch.nn.functional as nnFunc
from torch.autograd.variable import Variable
import gotex.utils as gu
import matplotlib.pyplot as plt
import time
import numpy as np
from os import mkdir
from os.path import isdir

class SemiDualOptimalTransportLayer(nn.Module):
    """
    forward(dualVariablePsi, inputDataX, targetDataY, batchSplitSize=None)
    """
    def __init__(self, targetDataY):
        super(SemiDualOptimalTransportLayer, self).__init__()
        self.targetDataY = targetDataY
        self.numTargetDataY = targetDataY.size(0)
        self.dualVariablePsi = nn.Parameter(torch.zeros(self.numTargetDataY))
        self.squaredY = torch.sum(self.targetDataY.transpose(0,1)**2,0,keepdim=True)

    def forward(self, inputDataX):
        loss = 0
        numInputDataX = inputDataX.size(0)
        dimInputDataX = inputDataX.size(1)
        batchSplitSize = numInputDataX 
        #if numInputDataX*dimInputDataX > 2**18:
         #   print('split')
        #    batchSplitSize = 2**18   
        InputDataX = torch.split(inputDataX, batchSplitSize)
        for x in InputDataX:            
            costMatrix = (torch.sum(x**2,1,keepdim=True) + self.squaredY - 2*torch.matmul(x,self.targetDataY.transpose(0,1)))/(2)
            loss += torch.sum(torch.min(costMatrix - self.dualVariablePsi.unsqueeze(0),1)[0])
        return loss/numInputDataX + torch.mean(self.dualVariablePsi)

def GotexVggTextureSynthesis(args):
    
    # get arguments from argparser
    target_img_path = args.target_image_path
    iter_max = args.iter_max
    iter_psi = args.iter_psi
    visu = args.visu
    save = args.save
    img_lr = args.img_lr
    psi_lr = args.psi_lr
    device = args.device

    # get target image
    target_img = gu.ReadImg(target_img_path)
    target_img = gu.PreProc(target_img)
    
    saving_folder = 'tmp/'
    
    if save:
        if not isdir(saving_folder):
            mkdir(saving_folder)
        gu.SaveImg(saving_folder+'original.png', gu.PostProc(target_img))
    
    # synthesized size
    if args.size is None:
        num_row = target_img.shape[2]
        num_col = target_img.shape[3]
    else:
        num_row = args.size[0]
        num_col = args.size[1]
        
    # visualize every 100 iteration when --visu is True
    monitoring_step = 50
        
    # initialize synthesized image
    target_mean = target_img.view(3,-1).mean(1).to(device) # 1x3x1
    target_std = target_img.view(3,-1).std(1).to(device) # 1x3x1
    synth_img = target_mean.view(1,3,1,1) + 0.05 * target_std.view(1,3,1,1) * torch.randn(1, 3, num_row,num_col, device=device)
    
    synth_img = Variable(synth_img, requires_grad=True)
    # synth_img = torch.randn(1,3, num_row, num_col, requires_grad=True, device=device)

    # initialize image optimizer
    image_optimizer = torch.optim.LBFGS([synth_img], lr=img_lr)

    # create VGG feature extractor (without padding for the target image)
    FeatExtractor = gu.CreateVggNet("VggModel", padding=False)
    # extract VGG features from the target_img
    input_features = FeatExtractor(target_img)
    # update normalizing of the VGG features
    norm_feat = [torch.sqrt(torch.sum(A.detach()**2,1).mean(0)) for A in input_features]
    FeatExtractor.normfeat = norm_feat
    # update input_features
    input_features = FeatExtractor(target_img)
    
    # create psi_optimizers and optimal transport layers
    psi_optimizers = []
    ot_layers  = []
    for i,feat in enumerate(input_features):
        ot_layers.append(SemiDualOptimalTransportLayer(feat.to(device)).to(device))
        psi_optimizers.append(torch.optim.ASGD(ot_layers[i].parameters(), lr=psi_lr, alpha=0.5))

    # create VGG feature extractor (without padding for the target image)
    FeatExtractor = gu.CreateVggNet("VggModel", padding=True).to(device)
    norm_feat_device = [n.to(device) for n in norm_feat]
    FeatExtractor.normfeat = norm_feat_device
    
    # initialize loss
    loss_list = [0]*iter_max
    starting_time = time.time()
    n_scales = len(norm_feat)
    
    # run optimization
    n_iter=[0]
    while n_iter[0] < iter_max:
        
        def closure():       
            # update dual variable psi            
            for itp in range(iter_psi):
                synth_features = FeatExtractor(synth_img)
                for i, feat in enumerate(synth_features):
                    psi_optimizers[i].zero_grad()
                    loss = -ot_layers[i](feat.detach())
                    loss.backward()
                    # normalize gradient
                    ot_layers[i].dualVariablePsi.grad.data /= ot_layers[i].dualVariablePsi.grad.data.norm()
                    psi_optimizers[i].step()  
       
            # update image
            # synth_features = FeatExtractor(synth_img)
            image_optimizer.zero_grad()
            loss = 0
            for i in range(n_scales):
                synth_features = FeatExtractor(synth_img)
                feat = synth_features[i]
                loss = ot_layers[i](feat)    
                loss.backward() 
            loss_list[n_iter[0]] = loss.item()
            
            # monitoring
            if ((n_iter[0]% monitoring_step) == 0):        
                elapsed_time = int(time.time()-starting_time)
                print('iteration = '+str(n_iter[0]))
                print('elapsed time = '+str(elapsed_time)+'s')
                print('OT loss = ' + str(loss.item()))
                if visu:
                    gu.ShowImg(gu.PostProc(synth_img))
                    plt.show()
                if save:
                    gu.SaveImg(saving_folder+'it'+str(n_iter[0])+'.png', gu.PostProc(synth_img.clone().detach()))
            

            n_iter[0]+=1
            return loss

        image_optimizer.step(closure)
      
    return synth_img, loss_list
