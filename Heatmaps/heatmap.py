# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 16:25:02 2022

@author: Nikhiel
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

import seaborn as sns





def pharray(df):
    
    num_sigs = len(df)
    m_adc = []
    timebin = []
    
    for i in range(30):
        
        s = np.concatenate((np.array(df.iloc[:, i]),
                             np.array(df.iloc[:, i+30]),
                             np.array(df.iloc[:, i+60])), axis=None).sum()
        m_adc.append(s/num_sigs)
        timebin.append(i)

    return timebin, m_adc







from matplotlib import colors




# load in the data 

df = pd.read_csv('Clean_simsigs_50k.csv')


L = df.shape[0]


# --------------------------------------------
# create average pulse height 

t, madc = pharray(df)
#---------------------------------------------


# generate time arrays repeated according to the number of rows in the df 
time = np.arange(0, 30)
x = np.tile(time, L)

# compute the individual pulse height array for each of the signals in our df

sigs = df.iloc[:, :90].to_numpy()


ph_sigs = sigs.reshape(L, 3, 30).sum(axis=1)

ex = np.arange(0, 30)
ey = np.arange(0, 260+1,5)

plt.figure(figsize=(10,6))
plt.hist2d(x, ph_sigs.flatten(), bins=[ex, ey], cmin=1, norm=colors.PowerNorm(1))
plt.plot(t, madc, marker='.', color='red', label='Average', linestyle='')
plt.colorbar(label="Counts")
plt.legend()
plt.title("Pulse Height Spectrum", fontsize=16)
plt.xlabel("Time bin (100ns)", fontsize=16)
plt.ylabel("ADC", fontsize=16)
plt.tight_layout()
plt.show()


'''
plt.figure(figsize=(10, 8))
sns.heatmap(np.histogram2d(x, ph_sigs.flatten(), bins=30, range=[[0,30],[0,500]]), cmap='viridis')
plt.show()

'''








