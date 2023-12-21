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
        self.conv1 = nn.Conv2d(self.channels,32,self.kernel_size,self.stride,self.padding)
        
        self.width = (self.width + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        self.depth = (self.depth + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        
        self.conv2 = nn.Conv2d(32,64,self.kernel_size,self.stride,self.padding)
        
        self.width = (self.width + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        self.depth = (self.depth + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        
        self.conv3 = nn.Conv2d(64,3,self.kernel_size,self.stride,self.padding)
        
        self.width = (self.width + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        self.depth = (self.depth + 2*self.padding - (self.kernel_size-1)-1)/self.stride + 1
        
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
        
        #self.fc1_3d_sub = nn.Linear(1,64)
        
        
        self.fc2 = nn.Linear(64,16)
        self.fc2_3d = nn.Linear(64,16)
        
        self.fc3 = nn.Linear(16,1)
        self.fc3_3d = nn.Linear(16,1)
        
        #self.fc4 = nn.Linear(16,1)
        #self.fc4_3d = nn.Linear(16,1)
        self.fc1_U = nn.Linear(1,16)
        self.fc2_U = nn.Linear(16,1)
        
        #self.pr_fc1 = nn.Linear(1,16)
        #self.pr_fc2 = nn.Linear(16,1)
        
        
        self.act = nn.ReLU()
        self.sig = nn.Sigmoid()
        
    def forward(self,Array,Input,U):
        U = U.unsqueeze(1)
        #Input1 = Input * PowerInput
        #Input1 = Input1.unsqueeze(dim = 0)
        #Input2 = self.sig(Input)
        
        
        #for i in range(len(Input2)):
            
        #    Input2[i] = 1/Input2[i] * U[i]
            
        #Input2 = Input2.unsqueeze(dim = 0)
        
        #Input = torch.cat((Input2,Input1),dim = 0)
        
        #Input = self.FCconcat1(Input.T)
        #Input = self.FCconcat2(Input)
        #Input = self.FCconcat3(Input)
        #Input = self.FCconcat4(Input)
        
        
        #Input = Input.T
        #Input = Input.squeeze()
        if self.use_Unet == True:
            
            return 0
        
        else:
            
            x1 = self.conv1(Input)
            x2 = self.conv2(x1)
            x3 = self.conv3(x2)
            #x3 = self.bn_2d(x3)#引入batch normalization
            x3 = self.act(x3)
            
            x1_3d = self.conv1_3d(Array)
            x2_3d = self.conv2_3d(x1_3d)
            x3_3d = self.conv3_3d(x2_3d)
            #x3_3d = self.bn_3d(x3_3d)#引入 batch normalization
            x3_3d = self.act(x3_3d)
            
            #x3_3d = self.fc1_3d(self.flatten(x3_3d))
            fc1_3d = self.fc1_3d(self.flatten(x3_3d))
            fc2_3d = self.fc2_3d(fc1_3d)
            fc4_3d = self.fc3_3d(fc2_3d)
            #fc4_3d = self.fc4_3d(fc3_3d)#生成间隙
            
            x5 = self.flatten(x3)
            #x3 = 1/x3
            
            #x4_3d = torch.Tensor(x3_3d.size()).cuda()
            x4_3d = fc4_3d*U/1.57e-5#生成雷诺数
            
            
            #x4_3d = self.flatten(x4_3d)
            
            fc1 = self.fc1(x5)
            fc2 = self.fc2(fc1)
            #fc3 = self.fc3(fc2)
            output1 = self.fc3(fc2)
            #output1 = self.act(output1)
            #Pr = torch.log10(torch.ones(output1.size()).cuda()*0.71)
            
            #output3 = self.pr_fc2(self.pr_fc1(Pr))
            
            #fc1_3d = self.fc1_3d_sub(x4_3d)
            #fc2_3d = self.fc2(fc1_3d)
            #fc3_3d = self.fc3(fc2_3d)
            Re1 = self.fc1_U(x4_3d)
            output2 = self.fc2_U(Re1)#生成雷诺数最终
            #output2 = self.act(output2)
            output = output2*output1*0.028#/1.57e-5
        
        return output, output2*output1
    
#Siamese Neural Network