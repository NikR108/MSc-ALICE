

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

#===================================================================


#df = pd.read_csv("Clean_simsigs_50k.csv")

# ==========================================================================================
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



# ========================================================================
# Heatmap Of PH for overall detector 

def phimg(df, name):

	L = len(df)
	tavg, avg = pharray(df)
	time = np.arange(0,30)
	x = np.tile(time, L)

	sigs = df.iloc[:, :90].to_numpy()
	phsigs = sigs.reshape(L, 3, 30).sum(axis=1)

	ex = np.arange(0, 30)
	ey = np.arange(0, 1.5*np.max(avg), 5)

	plt.figure(figsize=(12, 7))
	plt.title("Pulse Height Spectrum", fontsize=12)
	plt.xlabel("Time Bin (100ns)", fontsize=12)
	plt.ylabel("ADC", fontsize=12)
	plt.hist2d(x, phsigs.flatten(), bins=[ex, ey], cmin=1)
	plt.plot(tavg, avg, color='red', marker='.', label='Average', linestyle="")
	plt.colorbar()
	plt.legend()
	plt.tight_layout()
	plt.savefig("IMG_PH_{}.png".format(name))
	plt.show()





# Generates the Pulse Height heatmap per supermodule
# ----------------- >  requires df in csv format and the specific sm number in question 
#=================================================================================================

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
    plt.title("SM {}".format(sm), fontsize=6)
    plt.xlabel("Time Bin (100ns)", fontsize=6)
    plt.ylabel("ADC", fontsize=6)
    plt.hist2d(x, phsigs.flatten(), bins=[ex, ey], cmin=5)
    plt.plot(tavg, avg, color='red', marker='.', label='Average', linestyle='')
    plt.colorbar()
    plt.legend(fontsize=3)
    plt.tight_layout()
    #plt.savefig("IMG_PH.png")
    plt.show()


# Generates  Ph Heatmaps for supermodules 0 to 8 
def phsm08(df):
    
    plt.figure(figsize=(12,7))
    plt.subplot(3,3,1)
    phsm(df[df.sm==0], sm=0)

    plt.subplot(3,3,2)
    phsm(df[df.sm==1], sm=1)

    plt.subplot(3,3,3)
    phsm(df[df.sm==2], sm=2)

    plt.subplot(3,3,4)
    phsm(df[df.sm==3], sm=3)

    plt.subplot(3,3,5)
    phsm(df[df.sm==4], sm=4)

    plt.subplot(3,3,6)
    phsm(df[df.sm==5], sm=5)

    plt.subplot(3,3,7)
    phsm(df[df.sm==6], sm=6)

    plt.subplot(3,3,8)
    phsm(df[df.sm==7], sm=7)

    plt.subplot(3,3,9)
    phsm(df[df.sm==8], sm=8)

    plt.tight_layout()
    plt.show()



# Generates PH heatmaps for supermodules 9 to 17
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



#========================================
# Function produces the Average PH for a specific supermodule 

def avgphsm(df, s):
    
    #plt.figure(figsize=(10, 4))
    
    t, avg = pharray(df[df.sm==s])
    
    plt.plot(t, avg, marker='.', color='red')
    plt.title("SM {}".format(s), fontsize=10)
    plt.xlabel("Time bin (100ns)", fontsize=7)
    plt.ylabel("ADC", fontsize=7)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.show()

# Produces the avg PH for SMs 0 to 8
def avgphsm08(df):
    plt.figure(figsize=(12,8))

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


# Produces the Avg PH for supermodules 9 to 17
def avgphsm917(df):
    plt.figure(figsize=(12,8))

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


#=====================================================================
# Produces the average PH for all 18 supermodules in one plot grid 
def all_avg_phsm(df, name):
	'''
	plt.figure(figsize=(10,12))
	plt.subplot(3,3,1)
	avgphsm(df, s=0)
	etc......
	'''
	plt.figure(figsize=(10,12))
	for k in range(1, 19):
		plt.subplot(6,3,k)
		avgphsm(df, s=k-1)

	plt.tight_layout()
	plt.savefig("avgph_sm_{}.png".format(name))
	plt.show()


#====================================================================
# Produces the Heatmap PH for all 18 Supermodules in one plot grid 
def all_ph_sm(df, name): 

	'''
    plt.figure(figsize=(10,13))
    plt.subplot(6,3,1)
    phsm(df[df.sm==0], sm=0)
	etc......
	'''

	plt.figure(figsize=(10, 12))
	for k in range(1, 19):
		plt.subplot(6,3,k)
		phsm(df[df.sm==k-1], sm=k-1)

	plt.tight_layout()
	plt.savefig("PHimg_sm_{}.png".format(name))
	plt.show()





#==========================================
# DUAL COMPARISON OF AVG PULSE HEIGHTS 
#==========================================



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


# This function produces a 6 x 3 grid plot of dual comparisons of the avg ph curves of real and sim data 
def Dual_avgphsm(dfs, dfr):
    '''
    plt.figure(figsize=(10, 12))
    
    plt.subplot(6, 3, 1)
    avgphsm_dual(dfs, dfr, s=0)
    etc......
    '''

    plt.figure(figsize=(10,12))
    for k in range(1, 19):
    	plt.subplot(6,3,k)
    	avgphsm_dual(dfs, dfr, s=k-1)
    plt.tight_layout()
    plt.savefig("Dual_PH_sm.png")
    plt.show()



# =================================================
# Ratio of average PH for each SM 
#==================================================


def ratio_ph(dfs, dfr, s):
    
    #plt.figure(figsize=(10, 4))
    
    ts, avgs = pharray(dfs[dfs.sm==s])
    tr, avgr = pharray(dfr[dfr.sm==s])

    R = np.array(avgr)/np.array(avgs)
    
    plt.plot(ts, R, marker='.', color='seagreen')
    plt.axhline(y=1, linestyle='--', linewidth=1.4, color='k')
    plt.ylim(0.55, 1.45)
    #plt.plot(tr, avgr, marker='.', color='royalblue', label="Real")
    plt.title("SM {}".format(s), fontsize=12)
    plt.xlabel("Time bin (100ns)", fontsize=7)
    plt.ylabel("Real/Sim", fontsize=7)
    #plt.legend(fontsize=7)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()



   # This produces the ratio of avg PH curves for real and sim data 
   # --------==============--------------------==================--------========

def ratio_sm(dfs, dfr):
	'''
   	plt.figure(figsize=(10,12))

   	plt.subplot(6, 3, 1)
    ratio_ph(dfs, dfr, s=0)

    etc.........
	'''
	plt.figure(figsize=(10,12))
	for k in range(1, 19):
		plt.subplot(6,3,k)
		ratio_ph(dfs, dfr, s=k-1)

	plt.tight_layout()
	plt.savefig("Ratio_sm.png")
	plt.show()
	
    




