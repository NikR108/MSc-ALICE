# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 14:33:31 2022

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





def ratio_ph(dfs, dfr, s):
    
    #plt.figure(figsize=(10, 4))
    
    ts, avgs = pharray(dfs[dfs.sm==s])
    tr, avgr = pharray(dfr[dfr.sm==s])

    R = np.array(avgr)/np.array(avgs)
    
    plt.plot(ts, R, marker='.', color='mediumseagreen')
    plt.axhline(y=1, linestyle='--', linewidth=1.3 , color='k')
    #plt.plot(tr, avgr, marker='.', color='royalblue', label="Real")
    plt.title("SM {}".format(s), fontsize=12)
    plt.xlabel("Time bin (100ns)", fontsize=7)
    plt.ylabel("Real/Sim", fontsize=7)
    #plt.legend(fontsize=7)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.ylim(0.5, 1.5)
    plt.tight_layout()
    
    


def ratio_sm(dfs, dfr):

   plt.figure(figsize=(10,12))

   plt.subplot(6, 3, 1)
   ratio_ph(dfs, dfr, s=0)

   plt.subplot(6, 3, 2)
   ratio_ph(dfs, dfr, s=1)

   plt.subplot(6, 3, 3)
   ratio_ph(dfs, dfr, s=2)

   plt.subplot(6, 3, 4)
   ratio_ph(dfs, dfr, s=3)

   plt.subplot(6, 3, 5)
   ratio_ph(dfs, dfr, s=4)

   plt.subplot(6, 3, 6)
   ratio_ph(dfs, dfr, s=5)

   plt.subplot(6, 3, 7)
   ratio_ph(dfs, dfr, s=6)

   plt.subplot(6, 3, 8)
   ratio_ph(dfs, dfr, s=7)

   plt.subplot(6, 3, 9)
   ratio_ph(dfs, dfr, s=8)

   plt.subplot(6, 3, 10)
   ratio_ph(dfs, dfr, s=9)

   plt.subplot(6, 3, 11)
   ratio_ph(dfs, dfr, s=10)

   plt.subplot(6, 3, 12)
   ratio_ph(dfs, dfr, s=11)

   plt.subplot(6, 3, 13)
   ratio_ph(dfs, dfr, s=12)

   plt.subplot(6, 3, 14)
   ratio_ph(dfs, dfr, s=13)

   plt.subplot(6, 3, 15)
   ratio_ph(dfs, dfr, s=14)

   plt.subplot(6, 3, 16)
   ratio_ph(dfs, dfr, s=15)

   plt.subplot(6, 3, 17)
   ratio_ph(dfs, dfr, s=16)

   plt.subplot(6, 3, 18)
   ratio_ph(dfs, dfr, s=17)

   plt.tight_layout()
   plt.savefig("Ratio_sm.png")
   plt.show()
    
    
ratio_sm(dfs, dfr)


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

crit_val = chi2.ppf(q=0.95, df=29)
p_val = 1 - chi2.cdf(x=chi, df=29)

print("Chi Squared value = ", chi)
print("\nCritical value = ", crit_val)
print("\nP-value = ", p_val)







    
    