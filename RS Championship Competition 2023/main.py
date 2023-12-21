# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 00:10:47 2022

@author: Phoenix WANG, THE HONG KONG POLYTECHNIC UNIVERSITY Department of Mechanical Engineering
"""
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from Model import LSTM_Model
import torch
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from torch import optim
import torchvision
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



Salepath = './data/Sale_data.csv'
Quantitypath = './data/Quantantity.csv'
Raw_datapath = './data/Raw_Predata.csv'

Saledata = pd.read_csv(Salepath)
Quantitydata = pd.read_csv(Quantitypath)
Rawdata = pd.read_csv(Raw_datapath)

data = Saledata.values

X = torch.tensor(data)


train,test = train_test_split(X,test_size=0.33,random_state=42)
train = train
test = test

def createXY(dataset,n_past):
    dataX = []
    dataY = []
    temp = dataset.numpy()
    for i in range(n_past,len(dataset)):
        dataX.append(temp[i-n_past:i,0:temp.shape[1]])
        dataY.append(temp[i])
    dataXF = torch.tensor(np.array(dataX),dtype=torch.float32).cuda()
    dataYF = torch.tensor(np.array(dataY),dtype=torch.float32).cuda()
    return dataXF,dataYF

n_past = 3
trainX,trainY = createXY(train, n_past)
testX,testY = createXY(test, n_past)


model = LSTM_Model(4559,10,2).cuda()
learning_rate = 0.01
epoch = 10000
clip = 5.0
criterion = nn.MSELoss()
criterion = criterion.cuda()
optimizer =  optim.Adam(model.parameters(), lr=learning_rate)
Loss = []

for i in range(epoch):
    loss = torch.tensor(0,dtype=torch.float32).cuda()
    for idx in range(len(trainX)):
        output = model(trainX[idx])
        loss += criterion(output,trainY[idx])
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), clip)
    optimizer.step()
    if i % 100 == 0:
        loss_print = (loss.item()/trainX.shape[-1])/trainX.shape[0]
        Loss.append(loss_print)
        print(loss_print)

plt.figure()
plt.plot(Loss)
    