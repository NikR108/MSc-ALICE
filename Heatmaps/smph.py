# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 20:40:22 2022

@author: Nikhiel
"""




import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

from matplotlib import colors


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



def phsm(df, sm):
    
    # get length of df 
    L = len(df)
    
    
    
    # get the avg PH values 
    tavg, avg = pharray(df)
    
    # create 2d histogram 
    time = np.arange(0,30)
    x = np.tile(time, L)
    sigs = df.iloc[:, :90].to_numpy()
    phsigs = sigs.reshape(L, 3, 30).sum(axis=1)
    
    ex = np.arange(0, 30)
    ey = np.arange(0, 1.5*np.max(avg), 5)
    
    #H, xe, ye = np.histogram2d(x, phsigs.flatten(), bins=[ex,ey], range=[[0,30],[0,1.5*np.max(avg)]])
    
    
    #plt.figure(figsize=(12, 6))
    plt.title("SM {}".format(sm), fontsize=12)
    plt.xlabel("Time Bin (100ns)", fontsize=7)
    plt.ylabel("ADC", fontsize=7)
    plt.hist2d(x, phsigs.flatten(), bins=[ex, ey], cmin=5)
    plt.plot(tavg, avg, color='red', marker='.', label='Average', linestyle='')
    plt.colorbar()
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.legend(fontsize=7)
    #plt.savefig("IMG_PH.png")
    #plt.show()
    
    
#phsm(df[df.sm==0], sm=0)

def allsm(df): 
    plt.figure(figsize=(10,13))
    plt.subplot(6,3,1)
    phsm(df[df.sm==0], sm=0)

    plt.subplot(6,3,2)
    phsm(df[df.sm==1], sm=1)

    plt.subplot(6,3,3)
    phsm(df[df.sm==2], sm=2)

    plt.subplot(6,3,4)
    phsm(df[df.sm==3], sm=3)

    plt.subplot(6,3,5)
    phsm(df[df.sm==4], sm=4)

    plt.subplot(6,3,6)
    phsm(df[df.sm==5], sm=5)

    plt.subplot(6,3,7)
    phsm(df[df.sm==6], sm=6)

    plt.subplot(6,3,8)
    phsm(df[df.sm==7], sm=7)

    plt.subplot(6,3,9)
    phsm(df[df.sm==8], sm=8)


    plt.subplot(6,3,10)
    phsm(df[df.sm==9], sm=9)

    plt.subplot(6,3,11)
    phsm(df[df.sm==10], sm=10)

    plt.subplot(6,3,12)
    phsm(df[df.sm==11], sm=11)

    plt.subplot(6,3,13)
    phsm(df[df.sm==12], sm=12)

    plt.subplot(6,3,14)
    phsm(df[df.sm==13], sm=13)

    plt.subplot(6,3,15)
    phsm(df[df.sm==14], sm=14)

    plt.subplot(6,3,16)
    phsm(df[df.sm==15], sm=15)

    plt.subplot(6,3,17)
    phsm(df[df.sm==16], sm=16)

    plt.subplot(6,3,18)
    phsm(df[df.sm==17], sm=17)



    plt.tight_layout()
    plt.savefig("ph_img_sm.png")
    plt.show()


def phsm08(df):
    
    plt.figure(figsize=(12,7))
    #plt.subplot(3,3,1)
    #phsm(df[df.sm==0], sm=0)

    #plt.subplot(3,3,2)
    #phsm(df[df.sm==1], sm=1)

    #plt.subplot(3,3,3)
    #phsm(df[df.sm==2], sm=2)

    #plt.subplot(3,3,4)
    #phsm(df[df.sm==3], sm=3)

    #plt.subplot(3,3,5)
    #phsm(df[df.sm==4], sm=4)

    #plt.subplot(3,3,6)
    #phsm(df[df.sm==5], sm=5)

    #plt.subplot(3,3,7)
    #phsm(df[df.sm==6], sm=6)

    #plt.subplot(3,3,8)
    #phsm(df[df.sm==7], sm=7)

    #plt.subplot(3,3,9)
    #phsm(df[df.sm==8], sm=8)
    for k in range(1,10):
        plt.subplot(3,3,k)
        phsm(df[df.sm==k-1], sm=k-1)
    
    plt.tight_layout()
    plt.show()
    

def phsm917(df):
    
    plt.figure(figsize=(12,7))
    plt.subplot(3,3,1)
    phsm(df[df.sm==9], sm=9)

    plt.subplot(3,3,2)
    phsm(df[df.sm==10], sm=10)

    plt.subplot(3,3,3)
    phsm(df[df.sm==11], sm=11)

    plt.subplot(3,3,4)
    phsm(df[df.sm==12], sm=12)

    plt.subplot(3,3,5)
    phsm(df[df.sm==13], sm=13)

    plt.subplot(3,3,6)
    phsm(df[df.sm==14], sm=14)

    plt.subplot(3,3,7)
    phsm(df[df.sm==15], sm=15)

    plt.subplot(3,3,8)
    phsm(df[df.sm==16], sm=16)

    plt.subplot(3,3,9)
    phsm(df[df.sm==17], sm=17)

    plt.tight_layout()
    plt.show()
    

def gridplot(df):
    
    plt.figure(figsize=(10, 12))
    
    for k in range(1, 19):
        plt.subplot(6,3,k)
        phsm(df[df.sm==k-1], sm=k-1)
        
    plt.tight_layout()
    plt.savefig("Gridplot.png")
    plt.show()


#phsm08(dfr)
#phsm08(df)
#phsm917(dfr)

#phsm(df=dfr[dfr.sm==16], sm=16)
#allsm(df)


#import seaborn as sns
#sns.heatmap(dfr[dfr.sm==15].iloc[11,:90].to_numpy().reshape(3,30), cmap='viridis')

gridplot(df)
