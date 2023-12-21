# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 14:36:49 2022

@author: Phoenix WANG THE HONG KONG POLYTECHNIC UNIVERSITY MECHANICAL ENGINEERING
"""
import torch.nn as nn
from torch import optim
import torchvision



class LSTM_Model(nn.Module):
    def __init__(self,input_size,hidden_size,n_layers=2):
        super(LSTM_Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_size = input_size
        #self.embedding = nn.Embedding(input_size, hidden_size)
        self.LSTM = nn.LSTM(self.input_size,self.hidden_size,self.n_layers)
        self.fclayer = nn.Linear(self.hidden_size,self.output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,X):
        temp,_ = self.LSTM(X)
        temp2 = self.fclayer(temp)
        temp2 = self.relu(temp2)
        output = temp2.mean(axis=0)
        return output
    
