#Jake Williams and Samuel Tan

import torch
import pandas as pd
import random
import numpy as np
import os

'''This script produces the double dummy solver training data (ground truth) in sol100000.txt'''
np.random.seed(955)
dicto = {'2':0, '3':1, '4':2, '5':3, '6':4, '7':5, '8':6, '9':7, 'T':8, 'J':9, 'Q':10, 'K':11, 'A':12}

lines = [line.rstrip('\n') for line in open('sol100000.txt')]

indices = np.random.choice(100000, 20000, replace = False)
val_ctr = 0
trn_ctr = 0
for idx,l in enumerate(lines):
    inputs = torch.zeros(4,4,13)
    result = torch.zeros(1,4,13)
    l = l[:-21] + ' ' + l[-20:] #replace colon with space
    l = l.split()
    for i in range(4):
        tmp = l[i].split('.')
        for j in range(len(tmp)):
            for k in range(len(tmp[j])):
                inputs[i][j][dicto[tmp[j][k]]] = 1

    result = result.new_full((1,4, 13), int(l[4][3], 16))

    if idx in indices:
        #Note that this requires a folder called "val13" in the directory
        torch.save(torch.cat((inputs, result), 0), os.path.join('val13', str(val_ctr)+'.pt'))
        val_ctr +=1
    else:
        #Note that this requires a folder called "trn13" in the directory
        torch.save(torch.cat((inputs, result), 0), os.path.join('trn13', str(trn_ctr)+'.pt'))
        trn_ctr +=1



