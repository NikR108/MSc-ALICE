

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

from matplotlib import pyplot as plt, colors


# Load the dataframe 
df = pd.read_csv("Clean_realsigs_LHC22o.csv")

# get the length or number of rows ----> (no. of signals )
L = df.shape[0]

# Create a time array 
time = np.arange(0,30)

# Duplicate time array by number of signals present in our dataframe
x = np.tile(time, L)


# Get the signal arrays 
sigs = df.iloc[:, :90].to_numpy()

# Compute the individual PH for the sigs 
phsigs = sigs.reshape(L, 3, 30).sum(axis=1)

# Create the edges of the histogram
ex = np.arange(0, 30)
ey = np.arange(0, 250+1, 5)


# =======================================================================================
# Average pulse height of all the signals 

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

#=========================================================================================

tavg, avg = pharray(df)


H, xe, ye = np.histogram2d(x, phsigs.flatten(), bins=[ex,ey], range=[[0,30],[0,450]])


plt.figure(figsize=(12, 6))
plt.title("Pulse Height Spectrum", fontsize=16)
plt.xlabel("Time Bin (100ns)", fontsize=16)
plt.ylabel("ADC", fontsize=16)
plt.hist2d(x, phsigs.flatten(), bins=[ex, ey], cmin=1,  cmax=250000)
plt.plot(tavg, avg, color='red', marker='.', label='Average', linestyle='')
plt.colorbar()
plt.legend()
plt.tight_layout()
#plt.savefig("IMG_PH.png")
plt.show()


'''
plt.figure(figsize=(12, 6))
plt.imshow(H.T/30, cmap='viridis', extent=[xe[0], xe[-1], ye[-1], ye[0]], aspect='auto')
plt.plot(tavg, avg, marker='.', color='red', linestyle='', label='Average')
plt.ylim(ye[0], ye[-1])
plt.colorbar()
plt.legend()
plt.title("Pulse Height Spectrum", fontsize=15)
plt.xlabel("Time bin (100ns)", fontsize=15)
plt.ylabel("ADC", fontsize=15)
plt.tight_layout()
plt.show()
'''




