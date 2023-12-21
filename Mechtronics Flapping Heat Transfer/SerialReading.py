# -*- coding: utf-8 -*-
"""
Created on Mon May 30 17:13:22 2022

@author: PhoenixWANG
"""

import time
import serial
import pandas as pd
ser = serial.Serial(
    port = 'COM3',
    baudrate = 9600,
    parity = serial.PARITY_ODD,
    stopbits = serial.STOPBITS_TWO,
    bytesize = serial.SEVENBITS
    
    )

data = ''
list1,list2 = [],[]
count = 0
while count<60:
    data = ser.readline()
    t = time.time()
    ct = time.ctime(t)
    print(ct,':')
    print(data)
    #f = open('D:/test.txt','a')
    #f.writelines(data.decode('utf-8'))
    #f.close()
    datalist = data.decode('utf-8').split('\t')
    list1.append(float(datalist[0]))
    list2.append(float(datalist[1]))
    count += 1
diction = {
    'BasePanel':list1,
    'CoolingWings':list2
    }
outputpath = 'D:/result_default_Atomos.csv'
result = pd.DataFrame(diction)
result.to_csv(outputpath,sep = ',',index = False, header = True)
