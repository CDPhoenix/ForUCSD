# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 20:48:32 2022

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
import copy
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
    sigmoid = nn.Sigmoid()
    scaler = StandardScaler()
    temp = dataset.numpy()
    for i in range(n_past,len(dataset)):
        dataX.append(temp[i-n_past:i,0:temp.shape[1]])
        temp_copy = copy.deepcopy(temp[i])
        dataY.append(temp_copy)
    
    #change into one-hot code
    for data1 in dataY:
        length = data1.shape[0]
        for j in range(length):
            if data1[j] >0:
                data1[j] = 1
            else:
                data1[j] = 0
    
    #Normalize the data
    for i in range(len(dataX)):
        dataX[i] = scaler.fit_transform(dataX[i])
        dataX[i] = sigmoid(torch.tensor(dataX[i],dtype=torch.float32))
        dataX[i] = dataX[i].numpy()
    
    #Generate Tensor
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
        loss_print = loss.item()/train.shape[0]
        Loss.append(loss_print)
        print(loss_print)

plt.figure()
plt.plot(Loss)

