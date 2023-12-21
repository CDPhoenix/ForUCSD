# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:35:24 2023

@author: Phoenix WANG, Department of Mechanical Engineering, THE HONG KONG POLYTECHNIC UNIVERSITY

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
import optuna
from optuna.trial import TrialState
from matRead import MatRead
#from model import Model
from OriginalModel1 import Model
#from ModelBoundary import Model
from SHUFFLE import Shuffle
from OptunaTraining import Container
from torchmetrics.functional import r2_score
import sys
import time

# Define Process Bar:
# From: https://zhuanlan.zhihu.com/p/360444190


def process_bar(num, total):
    rate = float(num)/total
    ratenum = int(100*rate)
    r = '\r[{}{}]{}%'.format('*'*ratenum, ' '*(100-ratenum), ratenum)
    sys.stdout.write(r)
    sys.stdout.flush()


SEED = 6  # Proposed by Alice ZHAO

use_UNET = False  # True # Whether use UNet, True use, False not use
use_Optuna = False  # Whether use optuna to find optimal hyperparameters
optModel = False  # Whether use optuna to optimize network structure
use_CUDA = False
torch.manual_seed(SEED)

torch.cuda.manual_seed(SEED)

path = 'D:/PolyU/year3 sem02/URIS/COMSOL Practice/2DCases/2023.11.23/'

#cases = ['38_38_38_','38_76_38_','38_114_38_','38_114_76_','38_38_114_','38_76_114_','76_76_76_']

casepath = path + 'dataset1.mat'
datapath = path + 'data/'

cases = scio.loadmat(casepath)

cases = cases['dataset']
#cases = cases[0:175,:]
#cases = cases[: -2]


#exper = ['q','T']

#rowspaces = ['88','882','8858','76','762','7658']

Flatten = 0  # Whether flatten the data during the training and testing

rows = torch.tensor(cases[:, 3]).unsqueeze(1)

if use_UNET == True:

    batch_size = 3

else:

    batch_size = 3


batch_size_data = 110


Tensorcontainer, Arraycontainer = [], []

dataRead = MatRead(Tensorcontainer, Arraycontainer,
                   datapath, cases, batch_size, use_CUDA)

dataset = dataRead.datasetGenerate(Flatten)

#dataset = dataset.repeat(1,7,1,1)

#hc = dataRead.paramsRead('hc',target = 0)
#hc = dataRead.paramsRead('NU',target = 1)
NU = dataRead.paramsRead('hc', target=0)
U = dataRead.paramsRead('V', target=0)

Nu = dataRead.paramsRead('NU', target=0)
LSC = dataRead.paramsRead('LSC', target=0)
LSC_min = min(LSC.detach().numpy())
#H = Nu * 0.028/NU
#hc_expect = dataRead.paramsRead('Q_EXPECT',target = 0)
PowerInput = 1000


#NU = torch.cat((NU,hc_expect),dim=1)
NU = torch.cat((NU, LSC), dim=1)
#NU = torch.cat((NU,H),dim=1)
NU = torch.cat((NU, U), dim=1)
cases = torch.tensor(cases)
#NU1 = NU[0:4, :]
#NU2 = NU[5:, :]

HeightsWeight = torch.tensor([1, 2, 3]).float().unsqueeze(1)
#cases1 = cases[0:103,:]
#cases2 = cases[104:124,:]
#cases3 = cases[125:,:]
#cases = torch.cat((cases1,cases2),dim=0)
#cases = torch.cat((cases,cases3),dim=0)


#NU = torch.cat((NU1, NU2), dim=0)
NU = torch.cat((cases, NU), dim=1)


#A = dataset[4:,:,:,:]
#B = dataset[0:4,:,:,:]
#B = B.repeat(39,1,1,1)
#C = A-B+2
#dataset = torch.cat((B[0:4,:,:,:],C),dim=0)

dataset, NU = Shuffle(dataset, NU, SEED)

#bn = nn.BatchNorm2d(77)
#dataset = bn(dataset)
#NU1 = NU[0:103, :]
#NU2 = NU[104:124, :]
#NU3 = NU[125:, :]
#dataset1 = dataset[0:103, :]
#dataset2 = dataset[104:124, :]
#dataset3 = dataset[125:, :]


#NU = torch.cat((NU1, NU2), dim=0)
#NU = torch.cat((NU, NU3), dim=0)

#dataset = torch.cat((dataset1, dataset2), dim=0)
#dataset = torch.cat((dataset, dataset3), dim=0)

cases = NU[:, 0:5].cpu().detach().numpy()
rows = NU[:, 3].unsqueeze(1)
LSC = NU[:, 6]
#U = NU[:,-1]
#H = NU[:,-2]
back_up = NU[:, -1]

#hc_expect = NU[:,1]
#hc_expect = hc_expect.unsqueeze(1)

H = cases[:, 0:3]

H = torch.matmul(torch.tensor(H), HeightsWeight)/3
H = H.float()
rows = rows*back_up.unsqueeze(1)
back_up = torch.cat((back_up.unsqueeze(1), rows), dim=1)
back_up = torch.cat((back_up, H), dim=1)
Compare1 = LSC.detach().numpy()/LSC_min

NU = NU[:, 5]
NU = NU.unsqueeze(1)

#hc = hc.unsqueeze(1)


train_data_X = dataset[0:120, :, :, :]
train_data_Y = NU[0:120]
train_backup = back_up[0:120]


test_data_X = dataset[120:, :, :, :]
test_data_Y = NU[120:]
test_backup = back_up[120:]


#Reference_data_train = hc_expect[0:30]
#Reference_data = hc_expect[128:]

# Define Model and Hyperparameters

"""
train_data_X = torch.load('./train_data_X.pt')
train_data_Y = torch.load('./train_data_Y.pt')
train_backup = torch.load('./train_backup.pt')
test_data_X = torch.load('./test_data_X.pt')
test_data_Y = torch.load('./test_data_Y.pt')
test_backup = torch.load('./test_backup.pt')

if use_UNET == True:
    
    batch_size = 3
    
else:
    
    batch_size = 11
"""

batch_size_data = 120
model = Model(498, 35, 498, 35, batch_size, 1, 3, 1, 1, use_UNET).cuda()
criterion = nn.MSELoss()
criterion = criterion.cuda()
MAE = nn.L1Loss()  # Absolute Error

sizes = list(train_data_X.size())
sizes1 = list(test_data_X.size())

epoch = 100  # 220

mode = 1

model.load_state_dict(torch.load('./chosen_one_20231124_dict.pkl'))


if use_Optuna:

    number_of_trials = 15               # Number of Optuna trials

    container = Container(model, train_data_X, train_data_Y, train_backup,
                          test_data_X, test_data_Y, test_backup, batch_size, sizes, epoch, optModel=optModel)

    torch.manual_seed(SEED)

    torch.cuda.manual_seed(SEED)

    study = optuna.create_study(direction="minimize")
    study.optimize(container.objective, n_trials=number_of_trials)

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    # Find number of pruned and completed trials

    pruned_trials = study.get_trials(
        deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(
        deepcopy=False, states=[TrialState.COMPLETE])

    # Display the study statistics
    print("\nStudy statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    trial = study.best_trial
    print("Best trial:")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save results to csv file
    df = study.trials_dataframe().drop(
        ['datetime_start', 'datetime_complete', 'duration'], axis=1)  # Exclude columns
    # Keep only results that did not prune
    df = df.loc[df['state'] == 'COMPLETE']
    df = df.drop('state', axis=1)                 # Exclude state column
    df = df.sort_values('value')                  # Sort based on accuracy
    df.to_csv('optuna_results4.csv', index=False)  # Save to csv file

    # Display results in a dataframe
    print("\nOverall Results (ordered by accuracy):\n {}".format(df))

    # Find the most important hyperparameters
    most_important_parameters = optuna.importance.get_param_importances(
        study, target=None)

    # Display the most important hyperparameters
    print('\nMost important hyperparameters:')
    for key, value in most_important_parameters.items():
        print('  {}:{}{:.2f}%'.format(key, (15-len(key))*' ', value*100))


else:

    #learning_rate = 6.7778981540249974e-06
    learning_rate = 1e-3  # 1e-4#0.0007114426487435794
    clip = 5.0
    #optimizer =  optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer =  optim.RMSprop(model.parameters(), lr=learning_rate)
    Loss = []
    """
    if mode == 1:
        #定义生成权重的LSTM
        
        RNN = nn.LSTM(batch_size,1,2).cuda()
        hn = torch.randn(2,1,1).cuda()
        cn = torch.randn(2,1,1).cuda()
                
    elif mode == 0:
        #定义一步到位的LSTM
        RNN = nn.LSTM(3,1,3).cuda()
        hn = torch.randn(3,1).cuda()
        cn = torch.randn(3,1).cuda()
        
    else:
        
        raise RuntimeError('原神，启动！！！\n')
    
    """
    # Training

    model.train()

    # with torch.inference_mode():

    for j in range(epoch):

        # if j>100 & j<200:

       #     learning_rate = 1e-5#1e-4#0.0007114426487435794
       #     optimizer =  optim.Adam(model.parameters(), lr=learning_rate)

       # elif j>=200:

       #     learning_rate = 1e-5
       #     optimizer =  optim.Adam(model.parameters(), lr=learning_rate)

        final_loss = 0
        for z in range(int(sizes[0]/batch_size_data)):
            batch_data_X = train_data_X[z *batch_size_data:(z+1)*batch_size_data, :, :, :]
            batch_data_Y = train_data_Y[z *batch_size_data:(z+1)*batch_size_data, :]
            batch_data_backup = train_backup[z *batch_size_data:(z+1)*batch_size_data]

        # for i in range(int(sizes[1]/batch_size)):
            for i in range(int(sizes[1]/batch_size)):

                #data = train_data_X[:,i*batch_size:(i+1)*batch_size,:,:].cpu()
                epoch_now = j/100
                data = batch_data_X[:, i *batch_size:(i+1)*batch_size, :, :].cpu()
                data = data.cuda()
                #dataY = train_data_Y[i*batch_size:(i+1)*batch_size,:]
                Temp_array = data.unsqueeze(1).cpu()
                Temp_array = Temp_array.cuda()
                #data = data[:,0,:,:]
                #data = data.unsqueeze(1)
                #data = data.squeeze(1)
                #data = data.unsqueeze(0)
                optimizer.zero_grad()
                #output, x1, _, loss2, loss3 = model.forward(Temp_array, data, batch_data_backup.cuda(), 1)
                output,x1,_ = model.forward(Temp_array,data,batch_data_backup.cuda(),epoch_now)#,PowerInput)
                data = data.cpu()
                Temp_array = Temp_array.cpu()

                loss = torch.sqrt(criterion(output.cpu(), batch_data_Y))# + 2*torch.sqrt(loss2**2 + loss3**2)
                #loss2 = 1 - r2_score(output,dataY)

                #loss = loss1 * loss2

                #final_loss += loss.item()
                loss.backward()
                final_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                time.sleep(0.01)

                process_bar(i+1, int(sizes[1]/batch_size))

        if j % 1 == 0:
            loss_print = final_loss
            train_data_Y_mean = torch.mean(train_data_Y)
            rel_error = loss_print/train_data_Y_mean.item()
            Loss.append(loss_print)
            print('\n Training Loss of iteration {} is: '.format(j) + str(loss_print) + '\n')

    #x1 = train_data_X.unsqueeze(1)

    #ypredict1,x1 = model(x1,train_data_X[:,0,:,:].unsqueeze(1),train_backup)
    #rsq1 = r2_score(ypredict1,train_data_Y).item()
    # Loss.pop(0)

    plt.figure()
    plt.plot(Loss)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

    Error = []

    #length = int(sizes[1]/batch_size)

    data = train_data_X[:, i*batch_size:(i+1)*batch_size, :, :].cuda()
    Temp_array = data.unsqueeze(1)
    #output1, x1, _, _, _ = model.forward(Temp_array, data, train_backup.cuda(), 1)
    output1, x1, _ = model.forward(Temp_array, data, train_backup.cuda(), 1)

    fig1, ax1 = plt.subplots()
    #x1 = train_data_X.cpu().detach().numpy()
    ax1.plot(x1.cpu().detach().numpy(), train_data_Y.cpu().detach().numpy(), 'o', label='train_data')
    ax1.plot(x1.cpu().detach().numpy(),output.cpu().detach().numpy(), label='Prediction')
    #ax1.plot(x1,Reference_data_train.cpu().detach().numpy(),'o',label='Reference data')
    ax1.legend()
    ax1.set_title('Performance on train data')
    ax1.set_xlabel('Re$^m$/D')
    #ax1.set_ylabel('Coefficient of convective heat transfer')
    # ax1.set_ylabel('$\mathregular{log_{10}}$Q')
    ax1.set_ylabel('Coefficient of heat transfer')
    #ax1.text(25,200,'Absolute avg Error: ' + str(rel_error))
    plt.show()

    # Testing
    Error = []
    model.eval()

    with torch.no_grad():

        for i in range(int(sizes1[1]/batch_size)):
            data = test_data_X[:, i*batch_size:(i+1)*batch_size, :, :].cuda()
            #dataY = train_data_Y[i*2:(i+1)*2,:]
            Temp_array = data.unsqueeze(1)
            #data = data[:,0,:,:]
            #data = data.unsqueeze(1)
            #output, x2, _, loss2, loss3 = model.forward(Temp_array, data, test_backup.cuda(), 1)  # ,PowerInput)
            output,x2,_ = model.forward(Temp_array,data,test_backup.cuda(),1)#,PowerInput)
            #error = (output - test_data_Y)/test_data_Y
            loss = MAE(output, test_data_Y.cuda())/torch.mean(test_data_Y.cuda())
            Error.append(loss.cpu().detach().numpy())

        Error_avg = sum(Error)/len(Error)

        Error_numpyMean = np.mean(Error)
        print(Error_numpyMean)

        #x2 = test_data_X.unsqueeze(1)

        #ypredict2,x = model(x2,test_data_X[:,0,:,:].unsqueeze(1),test_backup)
        Rsq = r2_score(output, test_data_Y.cuda()).item()

    # Plotting the test dataset performance

    fig, ax = plt.subplots()
    #x = test_data_X.cpu().detach().numpy()
    ax.plot(x2.cpu().detach().numpy(), test_data_Y.cpu().detach().numpy(), 'o', label='test_data')
    ax.plot(x2.cpu().detach().numpy(),output.cpu().detach().numpy(), label='Prediction')
    #ax.plot(x,Reference_data.cpu().detach().numpy(),'o',label='Reference data')
    ax.legend()
    ax.set_title('Performance on test data')
    ax.set_xlabel('Re$^m$/D')
    #ax.set_ylabel('Coefficient of convective heat transfer')
    # ax.set_ylabel('$\mathregular{log_{10}}$Q')
    ax.set_ylabel('Coefficient of heat transfer')
    #ax.text(25,200,'Absolute avg Error: ' + str(Error_numpyMean))
    plt.show()

    data = dataset[:, i*batch_size:(i+1)*batch_size, :, :].cuda()
    Temp_array = data.unsqueeze(1)

#    output3, x3, ratio, _, _ = model.forward(Temp_array, data, back_up.cuda(), 1)
