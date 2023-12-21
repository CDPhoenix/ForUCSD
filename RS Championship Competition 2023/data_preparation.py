# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 19:32:12 2022

@author: Phoenix WANG - THE HONG KONG POLYTECHNIC UNIVERSITY
"""

import numpy as np
import pandas as pd
from collections import Counter

path = "./data/data.csv"

dataset = pd.read_csv(path)

#Drop the unuse labels
drop_labels = ['Invoice','Description','Customer ID','Country']
for i in drop_labels:
    dataset.drop(i,axis=1,inplace=True)

columns = list(dataset.columns.values)


#Transfer the date data into numerical data
for i in range(dataset.shape[0]):
    flag = []
    for j in range(len(dataset['InvoiceDate'][i])):
        if dataset['InvoiceDate'][i][j] == " ":#Don't consider time, day bias
            dataset['InvoiceDate'][i] = dataset['InvoiceDate'][i][:j+1]
            year = dataset['InvoiceDate'][i][:flag[0]] #Get the year
            month = dataset['InvoiceDate'][i][flag[0]+1:flag[1]] #Get the month
            date = dataset['InvoiceDate'][i][flag[1]+1:]# Get the date
            Date = int(year)*365 + int(month)*30 + int(date)# Calculate
            dataset['InvoiceDate'][i] = str(Date)#Reassign the date data
            break
        elif dataset['InvoiceDate'][i][j] == "/": #Get the position separate year,month and date
            flag.append(j)

#Calculate the sale feature    
dataset['Sale'] = dataset['Price']*dataset['Quantity']

#Sort value into time sequence
dataset = dataset.sort_values(by=['InvoiceDate', 'StockCode'])

#Get sum of the sale and quantity of a good in one day

Container = Counter(dataset['InvoiceDate'])
dataset_new = pd.DataFrame()

for date in Container:
    Slice = dataset.loc[dataset['InvoiceDate']==date]
    Container1 = Counter(Slice['StockCode'])
    Slice_input = pd.DataFrame()
    for code in Container1:
        data_temp = Slice.loc[Slice['StockCode']==code]
        quantity_sum = data_temp['Quantity'].sum()
        Sale_sum = data_temp['Sale'].sum()
        dataframe_temp = pd.DataFrame()
        dataframe_temp['InvoiceDate'] = [date]
        dataframe_temp['StockCode'] = [code]
        dataframe_temp['Quantity'] = [quantity_sum]
        dataframe_temp['Sale'] = [Sale_sum]
        Slice_input = pd.concat([Slice_input,dataframe_temp])
    dataset_new = pd.concat([dataset_new,Slice_input])

#Get the invoice date and stockcode of prepared data for forming tranning data
InvoiceDate_count = Counter(dataset_new['InvoiceDate'])
StockCode_count = Counter(dataset_new['StockCode'])

new_columns = []

for i in StockCode_count:
    new_columns.append(i)

zeros = np.zeros(len(StockCode_count))
count = 0

#Process data, date as index, stock as columns, the intersection values are the quantity or sale of this item on this day
for i in InvoiceDate_count:
    dataOfSale,dataOfQ = pd.DataFrame(zeros),pd.DataFrame(zeros)
    dataOfSale,dataOfQ = dataOfSale.T,dataOfQ.T
    dataOfSale.columns = new_columns
    dataOfQ.columns = new_columns
    dataOfSale.index = [i]
    dataOfQ.index = [i]
    tempSlice = dataset_new.loc[dataset_new['InvoiceDate']==i]
    tempCode = Counter(tempSlice['StockCode'])
    for j in tempCode:
        temp_row = tempSlice.loc[tempSlice['StockCode']==j]
        dataOfSale[j] = float(temp_row['Sale'])
        dataOfQ[j] = int(temp_row['Quantity'])
    if count == 0:
        count = count + 1
        Final_Sale = dataOfSale
        Final_Quantity = dataOfQ
    else:
        Final_Sale = pd.concat([Final_Sale,dataOfSale])
        Final_Quantity = pd.concat([Final_Quantity,dataOfQ])

#Save the data
Processed_rawDataoutput = './data/Raw_Predata.csv'        
Saleoutput = './data/Sale_data.csv'
Quantityoutput = './data/Quantantity.csv'

dataset_new.to_csv(Processed_rawDataoutput,sep=',',index=False,header=True)      
Final_Sale.to_csv(Saleoutput,sep=',',index=False,header=True)
Final_Quantity.to_csv(Quantityoutput,sep=',',index=False,header=True)        
        



