# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 16:31:15 2022

@author: Nikhiel
"""

import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt 
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




df = pd.read_csv("Clean_simsigs_50k.csv")

L = df.shape[0]

t = np.arange(0,30)

x = np.tile(t, df.shape[0])

sigs = df.iloc[:, :90].to_numpy()


ph_sigs = sigs.reshape(L, 3, 30).sum(axis=1)


edgey = np.arange(0, 300, 7)


H, xe, ye = np.histogram2d(x, ph_sigs.flatten(), bins=[t,edgey], range=[[0,30],[0,400]])


tavg, avg = pharray(df)


plt.figure(figsize=(12, 6))
#sns.heatmap(H, cmap='viridis')
plt.imshow(H.T, cmap='viridis', extent=[xe[0], xe[-1], ye[-1], ye[0]], aspect='auto')
plt.plot(tavg, avg, marker='.', color='red', linestyle='', label='Average')
plt.ylim(ye[0], ye[-1])
plt.colorbar()
plt.legend()
plt.title("Pulse Height Spectrum", fontsize=12)
plt.xlabel("Time bin (100ns)", fontsize=11)
plt.ylabel("ADC", fontsize=11)
plt.tight_layout()
plt.show()


from scipy.stats import chi2

def calc_chi(obs, exp):
    
    L = len(obs)
    vals = np.zeros(L)
    # get the length of the arrays 
    
    
    # loop over each element in the arrays and compute the squared diff divided by the expected
    
    for i in range(L):
        vals[i] = ((obs[i] - exp[i])**2)/(exp[i])
        
    chisq = vals.sum()
    
    return chisq


dfs = pd.read_csv("Clean_simsigs_50k.csv")
dfr = pd.read_csv("Clean_realsigs_LHC22o.csv")

t_s, a_s = pharray(dfs)
t_r, a_r = pharray(dfr)

calc_chi(obs = a_s, exp=a_r)


chi = calc_chi(obs = a_s, exp=a_r)

crit_val = chi2.ppf(q=0.95, df=30)
p_val = 1 - chi2.cdf(x=chi, df=30)

print("Chi Squared value = ", chi)
print("\nCritical value = ", crit_val)
print("\nP-value = ", p_val)




