# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:32:58 2023

@author: 86130
"""

import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ModelBoundary import Model
#from OriginalModel1 import Model
from torchmetrics.functional import r2_score
import matplotlib.pyplot as plt
from SHUFFLE import Shuffle

#cases = scio.loadmat('./dataset1.mat')
#cases = cases['dataset']

cases = torch.load('cases.pt')
#cases = cases.detach().numpy()
#ca0ses = cases[0:175,:]


#cases = Shuffle(cases,6)

#将张量转化为dataframe

def tensorToDF(dataset,use_CUDA):

    """

    生成的数据集有四列，每列数据含义如下：
    x: 横坐标值，雷诺数高幂除以2D特征长度
    hc: 原始数据的传热系数
    hc_pred：模型预测的传热系数
    vel: 对应的速度，一共只有三个速度值，2.3m/s, 3.6m/s, 3.9m/s

    """
    
    colnames = ['x','hc','hc_pred','vel','rowsXvel']

    if use_CUDA:
        
        dataset = dataset.cpu().detach().numpy()
    
    else:
        dataset = dataset.detach().numpy()
        
    dataset = pd.DataFrame(dataset)
    dataset.columns = colnames
    
    return dataset



def plotDrawing(dataset,vels):
    
    fig,ax = plt.subplots()
    
    for vel in vels:
        
        data = dataset[dataset['vel']==vel]
        x = data['x'].values
        hc = data['hc'].values
        #hc_pred = data['hc_pred'].values
        
        ax.plot(x,hc,'o',label = str(vel))
    return fig,ax

def output_estimation(h,beta):
    
    qsun = 1000
    P_stc = 206.9
    T_stc = 298.15
    T_ref = 300.15
    #q_rad = 0.87*5.6704*10**(-8)*(320.15**4-T_ref**4)
    q_refl = 400
    
    num = h*T_ref + qsun - 0-q_refl-P_stc*(1-beta*T_stc)
    den = h + beta*P_stc
    
    T_mod = num/den
    
    P = P_stc*(1 + beta*(T_mod - T_stc))
    
    return P

#是否用GPU
use_CUDA = False

batch_size = 3#batch_size大小，不要随便更改

#modelpath = './compare1_dict.pkl'#模型权重位置

modelpath = './Rsq0.951124.pkl'#模型权重位置
train_path = './train_dataset.pt'#训练集位置
test_path = './test_dataset.pt'#测试集数据位置
train_data_Xpath = './train_data_X.pt'#降维图像数据
test_data_Xpath = './test_data_X.pt'#降维图像数据

model = Model(498,35,498,35,batch_size,1,3,1,1)#初始化模型
model.load_state_dict(torch.load(modelpath))#加载训练好的权重
#model = torch.load(modelpath)

train_dataset = torch.load(train_path)
train_dataset = train_dataset.cpu()
test_dataset = torch.load(test_path)
test_dataset = test_dataset.cpu()
train_data_X = torch.load(train_data_Xpath)
test_data_X = torch.load(test_data_Xpath)

if use_CUDA:
    
    model = model.cuda()
    train_dataset = train_dataset.cuda()
    test_dataset = test_dataset.cuda()
    train_data_X = torch.load(train_data_Xpath).cuda()
    #t_data_X = test_data_X[120:150,:,:,:]
    test_data_X = torch.load(test_data_Xpath).cuda()

train_dataset_df = tensorToDF(train_dataset,use_CUDA)
test_dataset_df = tensorToDF(test_dataset,use_CUDA)
A = torch.cat((torch.tensor(train_dataset_df['hc']),torch.tensor(test_dataset_df['hc'])),dim=0)

vels = [1.5,2.3,3.6,3.9]
ax,fig = plt.subplots()
P1 = torch.tensor(output_estimation(train_dataset_df['hc_pred'].values, -0.0045))
P2 = torch.tensor(output_estimation(test_dataset_df['hc_pred'].values, -0.0045))
P3 = torch.tensor(output_estimation(train_dataset_df['hc'].values, -0.0045))
P4 = torch.tensor(output_estimation(test_dataset_df['hc'].values, -0.0045))
x1 = train_dataset[:,2]
x2 = test_dataset[:,2]
P = torch.cat((P1,P2),dim=0)
#P = P.detach().numpy()
P_original1 = torch.cat((P3,P4),dim=0)
#P_original = P_original1/torch.min(P_original1)
#P = P/torch.min(P_original1)
x = torch.cat((x1,x2),dim=0)

Evaluate = torch.cat((cases,P_original1.unsqueeze(1)),dim=1)#.detach().numpy()
Evaluate = torch.cat((Evaluate,x.unsqueeze(1)),dim=1)
Evaluate = torch.cat((Evaluate,P.unsqueeze(1)),dim=1).detach().numpy()
#Evaluate = torch.cat((cases,A.unsqueeze(1)),dim=1).detach().numpy()
Evaluation = pd.DataFrame(Evaluate)
Evaluation = Evaluation.sort_values(by=[0,1,2],ascending=True)
BaseCase = Evaluation[5][32]
P_1 = Evaluation[0:20].sort_values(by=[5],na_position='first')
P_2 = Evaluation[20:40].sort_values(by=[5],na_position='first')
P_3 = Evaluation[40:60].sort_values(by=[5],na_position='first')
P_4 = Evaluation[60:80].sort_values(by=[5],na_position='first')
P_5 = Evaluation[80:100].sort_values(by=[5],na_position='first')
P_6 = Evaluation[100:120].sort_values(by=[5],na_position='first')
P_7 = Evaluation[120:140].sort_values(by=[5],na_position='first')
P_8 = Evaluation[140:160].sort_values(by=[5],na_position='first')

Evaluation = Evaluation.sort_values(by=[5],na_position='first')

plt.figure(figsize=(20, 20))

baseline =np.ones((160,1),dtype=float)

fig,ax = plt.subplots()

ax.plot(Evaluation[6],Evaluation[7]/BaseCase,label = 'Prediction')


ax.plot(P_1[6],P_1[5]/BaseCase,'o',label = 'LLL')


ax.plot(P_2[6],P_2[5]/BaseCase,'*',label = 'LLM')


ax.plot(P_3[6],P_3[5]/BaseCase,'o',label = 'LLH')


ax.plot(P_4[6],P_4[5]/BaseCase,'*',label = 'LML')

ax.plot(P_5[6],P_5[5]/BaseCase,'o',label = 'LMH')

ax.plot(P_6[6],P_6[5]/BaseCase,'*',label = 'LHL')

ax.plot(P_7[6],P_7[5]/BaseCase,'o',label = 'LHM')

ax.plot(P_8[6],P_8[5]/BaseCase,'*',label = 'HHH')

ax.plot(Evaluation[6],baseline,'r',linestyle = '--')


ax.legend(loc = 4)


ax.set_title('Normalized power output VS h')


ax.set_xlabel('h [W m$^{-2}$K$^{-1}$]')


ax.set_ylabel('P/P$_{a}$')
plt.grid()

plt.show()

dataset = torch.cat((train_data_X,test_data_X),dim=0)
back_up = torch.cat((train_dataset[:,3:5],test_dataset[:,3:5]),dim=0)
#如何使用模型方式如下，对降维图像数据在 dimension1的方向上切片，然后升维

data = dataset[:,0:batch_size,:,:]
Temp_array = data.unsqueeze(1)
#back_up = train_dataset[:,-1]

output,x,_,_,_ = model.forward(Temp_array,data,back_up,1)



"""
画图任务：
    1. 画出 hc 与 hc_pred 对于 x 的图(plt.plot(x,y)),要求：hc的数据用一个又一个点画出，hc_pred用线画出
    2. 用不同的颜色的点标注出对应不同的速度值的hc
    

"""


fig,ax = plotDrawing(train_dataset_df,vels)
ax.plot(train_dataset_df['x'].values,train_dataset_df['hc_pred'],label = 'Prediction')
ax.legend()
ax.set_title('Performance on train data')
ax.set_xlabel('Re$^m$/D')
#ax1.set_ylabel('Coefficient of convective heat transfer')
#ax1.set_ylabel('$\mathregular{log_{10}}$Q')
ax.set_ylabel('Coefficient of heat transfer')
#ax1.text(25,200,'Absolute avg Error: ' + str(rel_error))
plt.show()


fig1,ax1 = plotDrawing(test_dataset_df,vels)
ax1.plot(test_dataset_df['x'].values,test_dataset_df['hc_pred'],label = 'Prediction')
ax1.legend()
ax1.set_title('Performance on test data')
ax1.set_xlabel('Re$^m$/D')
#ax1.set_ylabel('Coefficient of convective heat transfer')
#ax1.set_ylabel('$\mathregular{log_{10}}$Q')
ax1.set_ylabel('Coefficient of heat transfer')
#ax1.text(25,200,'Absolute avg Error: ' + str(rel_error))
plt.show()

A = np.array(test_dataset_df['hc'])

B = np.array(test_dataset_df['hc_pred'])

rel_error = np.mean(abs(A-B))/np.mean(B)
