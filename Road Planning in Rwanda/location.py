# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 23:46:39 2023

@author: Phoenix WANG, Department of Mechanical Engineering, THE HONG KONG POLYTECHNIC UNIVERSITY
"""

import pandas as pd
import numpy as np

from collections import Counter

data = pd.read_csv('./Mapping_Overall.csv')

data['Y'] = -1*data['Y']

data = data.drop(columns = ['description'])

Village = []




for i in range(len(data)):
    
    temp = data['Name'][i]
    
    string = []
    
    for j in range(len(temp)):
        
        if temp[j] == ' ':
            
            break
        
        else:
            
            string.append(temp[j])
    
    Village.append(''.join(string))
    
    
data['village'] = Village

data.sort_values(['village'],inplace = True)

villagenames = Counter(Village)

Village_type = list(villagenames.keys())



selected_houses = []

for villagename in Village_type:


    #villagename = 'Kajevuba' 
    
    temp = data[data['village'] == villagename]
    
    temp['absdistance'] = np.sqrt(temp['X']**2 + temp['Y']**2)
    
    housenames = temp['Name']
    
    container = []
    
    for i in temp['absdistance']:
        
        target = i
        
        Newdata = abs(temp['absdistance'] - target)
    
        container.append(Newdata)
        
    
    final_data = pd.DataFrame(container)
    final_data.index = housenames
    final_data.columns = housenames
    
    comparison = list(sum(final_data.values))
    
    target_index = comparison.index(min(comparison))
    
    selected_house = list(housenames)[target_index]
    
    selected_houses.append(selected_house)

















    