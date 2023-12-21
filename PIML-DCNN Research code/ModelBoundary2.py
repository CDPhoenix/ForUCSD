# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:14:54 2023

@author: 86130
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision
import numpy as np
import pandas as pd
from UNET import UNet
class Model(nn.Module):
    def __init__(self,width,depth,width3d,depth3d,batch_size,channels_3d,kernel_size,stride,padding,use_Unet = False):
        super(Model, self).__init__()
        self.width = width
        self.depth = depth
        self.width3d = width3d
        self.depth3d = depth3d
        self.height3d = batch_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.channels = batch_size
        self.channels_3d = channels_3d
        self.use_Unet = use_Unet
        #Setting 2D convolution
        
        self.conv = 
    def forward(self,Array,Input,back_up):
        
        U = back_up.unsqueeze(1)
        
        U1 = U/torch.max(U)
        
        #for i in range(len(Array)):
            #Array[i,:,:,:] = Array[i,:,:,:]+U1[i,:]

        
        if self.use_Unet == True:
            
            return 0
        
        else:
            
            x1 = self.conv1(Input)
            x2 = F.relu(F.max_pool2d(x1,2))
            x3 = self.conv2(x2)
            x4 = F.relu(F.max_pool2d(x3,2))
            #x3 = self.conv3(x2)
            #x3 = self.act(x3)
            
            
            
            x1_3d = self.conv1_3d(Array)
            x2_3d = F.relu(F.max_pool3d(x1_3d,2))
            #x3_3d = self.conv3_3d(x2_3d)
            #x3_3d = self.act(x3_3d)
            LSC = self.fc1_3d(self.flatten(x2_3d))
            #LSC = F.dropout(self.fc1_3d(self.flatten(x2_3d)),training=self.training)
            LSC = self.fc2_3d(LSC)
            #c4_3d = self.act2(self.fc3_3d(fc2_3d))     
            
            #F.dropout(x, training=self.training)
            x5 = self.flatten(x4)

            
            
          
            Re = LSC*U/1.57e-5#生成雷诺数
            
            #ratio = fc4_3d.cpu().detach().numpy()/min(fc4_3d.cpu().detach().numpy())
            ratio = 0
            A = LSC.cpu().detach().numpy()

            
            #fc1 = F.dropout(self.fc1(x5),training=self.training)
            fc1 = self.fc1(x5)
            D = self.fc2(fc1)
            #fc3 = self.fc3(fc2)
            #output1 = self.act2(fc3)#*6.21#4.81

            B = D.cpu().detach().numpy()
            C = Re.cpu().detach().numpy()
            
            
            #weigths1 = self.conv_weights1(Array)
            #weigths2 = self.conv_weights2(weigths1)
            #weights = self.lin_weights(self.flatten(weigths2))
            
            
            Re1 = self.fc1_U(Re)
            output2 = self.fc2_U(Re1)#/100#生成雷诺数最终

            output = output2/D*0.028#/1.57e-5
            #output = weights*output2*0.028#/1.57e-5
            E = output2.cpu().detach().numpy()

        return output,output2/D,ratio,torch.mean(LSC)/8 + 4/torch.mean(LSC),torch.mean(D)/6 + 3.17/torch.mean(D)