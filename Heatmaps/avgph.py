# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 14:30:56 2022

@author: Nikhiel
"""

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

from matplotlib import pyplot as plt, colors


dfs = pd.read_csv("Clean_simsigs_50k.csv")
dfr = pd.read_csv("Clean_realsigs_LHC22o.csv")


# this compute the avg ph information 
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


#----------------------------------------------------------------------------
# This function computes the dual Avg PH for a specified SM
def avgphsm_dual(dfs, dfr, s):
    
    #plt.figure(figsize=(10, 4))
    
    ts, avgs = pharray(dfs[dfs.sm==s])
    tr, avgr = pharray(dfr[dfr.sm==s])
    
    plt.plot(ts, avgs, marker='.', color='crimson', label="Sim")
    plt.plot(tr, avgr, marker='.', color='royalblue', label="Real")
    plt.title("SM {}".format(s), fontsize=12)
    plt.xlabel("Time bin (100ns)", fontsize=7)
    plt.ylabel("ADC", fontsize=7)
    plt.legend(fontsize=7)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    #plt.show()
    
    
    
    
    
def Dual_avgphsm(dfs, dfr):
    
    plt.figure(figsize=(10, 12))
    
    plt.subplot(6, 3, 1)
    avgphsm_dual(dfs, dfr, s=0)
    
    plt.subplot(6, 3, 2)
    avgphsm_dual(dfs, dfr, s=1)
    
    plt.subplot(6, 3, 3)
    avgphsm_dual(dfs, dfr, s=2)
    
    plt.subplot(6, 3, 4)
    avgphsm_dual(dfs, dfr, s=3)
    
    plt.subplot(6, 3, 5)
    avgphsm_dual(dfs, dfr, s=4)
    
    plt.subplot(6, 3, 6)
    avgphsm_dual(dfs, dfr, s=5)
    
    plt.subplot(6, 3, 7)
    avgphsm_dual(dfs, dfr, s=6)
    
    plt.subplot(6, 3, 8)
    avgphsm_dual(dfs, dfr, s=7)
    
    plt.subplot(6, 3, 9)
    avgphsm_dual(dfs, dfr, s=8)
    
    plt.subplot(6, 3, 10)
    avgphsm_dual(dfs, dfr, s=9)
    
    plt.subplot(6, 3, 11)
    avgphsm_dual(dfs, dfr, s=10)
    
    plt.subplot(6, 3, 12)
    avgphsm_dual(dfs, dfr, s=11)
    
    plt.subplot(6, 3, 13)
    avgphsm_dual(dfs, dfr, s=12)
    
    plt.subplot(6, 3, 14)
    avgphsm_dual(dfs, dfr, s=13)
    
    plt.subplot(6, 3, 15)
    avgphsm_dual(dfs, dfr, s=14)
    
    plt.subplot(6, 3, 16)
    avgphsm_dual(dfs, dfr, s=15)
    
    plt.subplot(6, 3, 17)
    avgphsm_dual(dfs, dfr, s=16)
    
    plt.subplot(6, 3, 18)
    avgphsm_dual(dfs, dfr, s=17)
    
    plt.tight_layout()
    plt.savefig("Avg_PH_sm.png")
    plt.show()
    
    
    
#Dual_avgphsm(dfs, dfr)
 
def gplot(dfs, dfr):
    
    plt.figure(figsize=(10, 12))
    
    for k in range(1, 19):
        plt.subplot(6,3,k)
        avgphsm_dual(dfs, dfr, s=k-1)
        
    plt.tight_layout()
    plt.savefig("Gplot.png")
    plt.show()


   