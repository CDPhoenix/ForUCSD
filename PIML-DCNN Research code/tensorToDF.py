# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:43:00 2023

@author: 86130
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision
import numpy as np
import pandas as pd

def tensorTodf(dataset):
    
    colnames = ['x','hc','hc_pred','vel']
    
    
    
    return 0