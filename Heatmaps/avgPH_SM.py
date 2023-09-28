# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 13:43:46 2022

@author: Nikhiel
"""

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

from matplotlib import pyplot as plt, colors


df = pd.read_csv("Clean_simsigs_50k.csv")
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






def avgphsm(df, s):
    
    #plt.figure(figsize=(10, 4))
    
    t, avg = pharray(df[df.sm==s])
    
    plt.plot(t, avg, marker='.', color='red')
    plt.title("SM {}".format(s), fontsize=10)
    plt.xlabel("Time bin (100ns)", fontsize=10)
    plt.ylabel("ADC", fontsize=10)
    plt.show()
    
    
    
    
def avgphsm08(df):
    plt.figure(figsize=(12,7))

    plt.subplot(3,3,1)
    avgphsm(df, s=0)    

    plt.subplot(3,3,2)
    avgphsm(df, s=1)

    plt.subplot(3,3,3)
    avgphsm(df, s=2)

    plt.subplot(3,3,4)
    avgphsm(df, s=3)

    plt.subplot(3,3,5)
    avgphsm(df, s=4) 

    plt.subplot(3,3,6)
    avgphsm(df, s=5)

    plt.subplot(3,3,7)
    avgphsm(df, s=6)

    plt.subplot(3,3,8)
    avgphsm(df, s=7)

    plt.subplot(3,3,9)
    avgphsm(df, s=8)
    plt.tight_layout()

    plt.show()       
    


def avgphsm917(df):
    plt.figure(figsize=(12,7))

    plt.subplot(3,3,1)
    avgphsm(df, s=9)    

    plt.subplot(3,3,2)
    avgphsm(df, s=10)

    plt.subplot(3,3,3)
    avgphsm(df, s=11)

    plt.subplot(3,3,4)
    avgphsm(df, s=12)

    plt.subplot(3,3,5)
    avgphsm(df, s=13) 

    plt.subplot(3,3,6)
    avgphsm(df, s=14)

    plt.subplot(3,3,7)
    avgphsm(df, s=15)

    plt.subplot(3,3,8)
    avgphsm(df, s=16)

    plt.subplot(3,3,9)
    avgphsm(df, s=17)
    plt.tight_layout()

    plt.show()       



#avgphsm08(dfr)
avgphsm08(df)

avgphsm917(df)


avgphsm08(dfr)
avgphsm917(dfr)




