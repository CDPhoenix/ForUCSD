# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:38:32 2023

@author: Phoenix WANG, Department of Mechanical Engineering, THE HONG KONG POLYTECHNIC UNIVERSITY
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
    def __init__(self,width,depth,width3d,depth3d,height3d,batch_size,channels_3d,
                 kernel_size,stride,padding,use_Unet = False):
        
        
        super(Model, self).__init__()
        self.width = width
        self.depth = depth
        self.width3d = width3d
        self.depth3d = depth3d
        #self.height3d = batch_size
        self.height3d = height3d
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.channels = batch_size
        self.channels_3d = channels_3d
        self.use_Unet = use_Unet
        
        
        #Setting 2D convolution  
        
        self.conv1 = nn.Conv2d(1,32,self.kernel_size,self.stride,self.padding)
        
        self.width = (self.width + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        self.depth = (self.depth + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        
        self.conv2 = nn.Conv2d(32,64,self.kernel_size,self.stride,self.padding)
        
        self.width = (self.width + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        self.depth = (self.depth + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        
        self.conv3 = nn.Conv2d(64,3,self.kernel_size,self.stride,self.padding)
        
        #self.width = (self.width + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        #self.depth = (self.depth + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        
        
        
        #Setting 3D convolution
        
        self.conv1_3d = nn.Conv3d(self.channels_3d,3,self.kernel_size,self.stride,self.padding)
        self.width3d = (self.width3d + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        self.depth3d = (self.depth3d + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        self.height3d = (self.height3d + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        
        self.conv2_3d = nn.Conv3d(3,16,self.kernel_size,self.stride,self.padding)
        self.width3d = (self.width3d + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        self.depth3d = (self.depth3d + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        self.height3d = (self.height3d + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        
        self.conv3_3d = nn.Conv3d(16,3,self.kernel_size,self.stride,self.padding)
        self.width3d = (self.width3d + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        self.depth3d = (self.depth3d + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        self.height3d = (self.height3d + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        
        
        
        self.bn_2d = nn.BatchNorm2d(3)
        self.bn_3d = nn.BatchNorm3d(3)
        
        
        #self.FCconcat1 = nn.Linear(2,16)
        #self.FCconcat2 = nn.Linear(16,32)
        #self.FCconcat3 = nn.Linear(32,16)
        #self.FCconcat4 = nn.Linear(16,1)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(int(self.width*self.depth*3),64)
        self.fc1_3d = nn.Linear(int(self.width3d*self.depth3d*self.height3d*3),64)
        
        
        self.fc2 = nn.Linear(64,32)
        self.fc2_3d = nn.Linear(64,32)
        
        self.fc3 = nn.Linear(32,16)
        self.fc3_3d = nn.Linear(32,16)
        
        self.fc4 = nn.Linear(16,1)
        self.fc4_3d = nn.Linear(16,1)
        #self.fc1_U = nn.Linear(1,14)
        #self.fc2_U = nn.Linear(14,1)
        
        self.weights = nn.Linear(2,1)
        self.act = nn.ReLU()
        self.sig = nn.Sigmoid()
        
    def forward(self,Array,Input,U):#,PowerInput):
        
        #2D Convolution for H
        
        if self.use_Unet == True:
            
            self.UNet = UNet(self.channels).cuda()
            x3 = self.UNet(Input)
        """    
        
        else:
            
            x1 = self.conv1(Input)
            x2 = self.conv2(x1)
            x3 = self.conv3(x2)
            #x3 = self.bn_2d(x3)#引入batch normalization
            x3 = self.act(x3)
        """
        #3D Convolution for L
        
        x1_3d = self.conv1_3d(Array)
        x2_3d = self.conv2_3d(x1_3d)
        x3_3d = self.conv3_3d(x2_3d)
        #x3_3d = self.bn_3d(x3_3d)#引入 batch normalization
        x3_3d = self.bn_3d(x3_3d)
        x3_3d = self.act(x3_3d)
            
            
        #x5 = self.flatten(x3)
        #x3 = 1/x3
            
        #x4_3d = torch.Tensor(x3_3d.size()).cuda()
        x1_2d = Input
        x2_2d = self.conv1(x1_2d)
        x3_2d = self.conv2(x2_2d)
        x4_2d = self.conv3(x3_2d)
        
        x4_2d = self.bn_2d(x4_2d)
        x4_2d = self.act(x4_2d)
        """
        x4_3d = torch.zeros(x3_3d.size()).cuda()
            
        if list(U.size()) == []:
            
            x4_3d = x3_3d*U
            
        else:
            
            for i in range(len(x4_3d)):
                
                x4_3d[i] = x3_3d[i]*U[i]
        """        
        x4_3d = self.flatten(x3_3d)
        x4_2d = self.flatten(x4_2d)
        """    
        fc1 = self.fc1(x5)
        fc2 = self.fc2(fc1)
        fc3 = self.fc3(fc2)
        output1 = self.fc4(fc3)
        #output1 = self.act(output1)
        """    
        fc1_3d = self.fc1_3d(x4_3d)
        fc2_3d = self.fc2_3d(fc1_3d)
        fc3_3d = self.fc3_3d(fc2_3d)
        
        
        fc1 = self.fc1(x4_2d)
        fc2 = self.fc2(fc1)
        fc3 = self.fc3(fc2)
        
        D = self.fc4(fc3)#characteristic height(in log10)
        La = self.fc4_3d(fc3_3d)#characteritic length (in log10)
        
        Re = La + np.log10(2.3/1.57e-5)
        
        lamda = D + np.log10(0.028/40)
        
        Pr = torch.log10(torch.ones(Re.size()).cuda()*0.71)
        #bias = torch.ones(Re.size()).cuda()
        Input_final = torch.cat((Re,Pr),dim=1)
        #Input = torch.cat((Input,bias),dim=1).cuda()
        output = self.weights(Input_final)
        output = output - lamda
        
        x_axis = La*self.weights.state_dict()['weight'][0][0]-D#(10**La)**self.weights.state_dict()['weight'][0][0]/(10**D)
        #output2 = self.act(output2)
        #output = output2*output1*0.028/1.57e-5
        
        return output,x_axis
    
