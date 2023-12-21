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
        
        #self.fc1_power = nn.Linear(64,16)
        #self.fc2_power = nn.Linear(16,1)
        #self.fc4 = nn.Linear(16,1)
        #self.fc4_3d = nn.Linear(16,1)
        
        self.fc1_U = nn.Linear(1,16)
        self.fc2_U = nn.Linear(16,1)
        
        #self.pr_fc1 = nn.Linear(1,16)
        #self.pr_fc2 = nn.Linear(16,1)
        #self.D_thru = nn.Linear(1,1)
        #self.Re_thru = nn.Linear(1,1)
        
        
        #self.act1 = nn.Softmax(dim=0)
        self.act = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self,Array,Input,back_up):
        
        U = back_up.unsqueeze(1)
        #H = back_up[:,-2].unsqueeze(1)
        #LSC = back_up[:,0]
        
        
        
        
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
            
            
            #for i in range(len(temp_2d)):
                
                #temp_2d[i,:,:,:] = x3[i,:,:,:]/H[i]
            
            x1_3d = self.conv1_3d(Array)
            x2_3d = self.conv2_3d(x1_3d)
            x3_3d = self.conv3_3d(x2_3d)
            x3_3d = self.act(x3_3d)
            
            #temp_3d = torch.zeros(x3_3d.size()).cuda()
            
            #for i in range(len(temp_3d)):
                
            #    temp_3d[i,:,:,:,:] = x3_3d[i,:,:,:,:]*LSC[i]
                
            #x3_3d = self.bn_3d(x3_3d)#引入 batch normalization
    
            
            #x3_3d = self.fc1_3d(self.flatten(x3_3d))
            fc1_3d = self.fc1_3d(self.flatten(x3_3d))
            fc2_3d = self.fc2_3d(fc1_3d)
            #fc4_3d = fc2_3d/torch.mean(fc2_3d)
            #fc3_3d = self.fc3_3d(fc2_3d)
            fc4_3d = self.fc3_3d(fc2_3d)      
            
            #power1 = self.fc1_power(fc1_3d)
            #power2 = self.sig(self.fc2_power(power1))
            #fc4_3d = fc2_3d#生成间隙
            
            x5 = self.flatten(x3)
            #x3 = 1/x3
            
            #x4_3d = torch.Tensor(x3_3d.size()).cuda()
            #x4_3d = torch.zeros(fc4_3d.size()).cuda()
            
            
          
            x4_3d = fc4_3d*U/1.57e-5#生成雷诺数
            
            ratio = fc4_3d.cpu().detach().numpy()/min(fc4_3d.cpu().detach().numpy())
            A = fc4_3d.cpu().detach().numpy()
            
            #temp1 = self.Re_thru(x4_3d)
            
            #x4_3d = self.flatten(x4_3d)
            
            fc1 = self.fc1(x5)
            fc2 = self.fc2(fc1)
            #fc3 = fc2/torch.mean(fc2)
            #fc3 = self.fc3(fc2)
            output1 = self.fc3(fc2)
            #fc4 = fc2/torch.mean(fc)
            #output1 = torch.zeros(temp1.size()).cuda()
            B = output1.cpu().detach().numpy()
            C = x4_3d.cpu().detach().numpy()
            
            
            Re1 = self.fc1_U(x4_3d)
            output2 = self.fc2_U(Re1)/100#生成雷诺数最终
            #output2 = self.act(output2)
            output = output2/output1*0.028#/1.57e-5
            D = output2.cpu().detach().numpy()
            #output = output2/output1*0.028#/1.57e-5
        return output, output2/output1,ratio
    
#Siamese Neural Network