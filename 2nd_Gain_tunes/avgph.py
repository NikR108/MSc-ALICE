
# Load packages
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from scipy.stats import chisquare 

# Avg pulse height info as an array 

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


 #--------------------------------------------------------------------


# Now create a function to plot the results 

def ph_plot(df, fname, cp="navy", cl="royalblue"):

	timebin, adc = pharray(df)

	plt.figure(figsize=(15, 7))
	plt.scatter(timebin, adc, color=cp)
	plt.plot(timebin, adc, color=cl)
	plt.xlabel("Time bins (100ns)", fontsize=14)
	plt.ylabel("Average ADC", fontsize=14)
	plt.title("Average Pulse Height", fontsize=14)
	plt.tight_layout()
	plt.savefig("ph_{}.png".format(fname))
	plt.show()


#------------------------------------------------------------------------


# Now create a function for dual plots of real and sim pulse heights 

def dual_ph(rdf, sdf, fname, c1="navy", c2="royalblue", c3='coral', c4='red'):

	time_r, adc_r = pharray(rdf)
	time_s, adc_s = pharray(sdf)

	#chi = np.round(chisquare(f_obs=np.array(adc_r)/np.array(adc_r).sum(),
	 #f_exp = np.array(adc_s)/np.array(adc_s).sum()),5)

	plt.figure(figsize=(15, 7))
	plt.scatter(time_r, adc_r, color=c1)
	plt.plot(time_r, adc_r, color=c2, label="Real")
	plt.scatter(time_s, adc_s, color=c3)
	plt.plot(time_s, adc_s, color=c4, label="Sim")
	plt.legend()
	#plt.text(25, 170, "$\chi^2\, / \, $ ndof = {} / {}".format(chi[0], len(adc_s)))
	#plt.text(25, 165, "p-value = {}".format(chi[1]))
	plt.xlabel("Time bin (100ns)", fontsize=14)
	plt.ylabel("Average ADC", fontsize=14)
	plt.title("Average Pulse Height", fontsize=14)
	plt.tight_layout()
	plt.savefig("ph_dual_{}.png".format(fname))
	plt.show()

#-------------------------------------------------------------------------


# create a function to plot the ratio of real and simulated pulse heights 

def ph_ratio(rdf, sdf, fname, c1="seagreen", c2="green"):

	time_r, adc_r = pharray(rdf)
	time_s, adc_S = pharray(sdf)
	#ratio = []
	# take the ratio, divide real over sim 
	#for i in range(len(adc_r)):
		#ratio.append(np.array(adc_r[i])/np.array(adc_S[i]))
	
	ratio = np.array(adc_r)/np.array(adc_S)
	

	plt.figure(figsize=(15, 7))
	plt.scatter(time_r, ratio, color=c1)
	plt.plot(time_r, ratio, color=c2)
	plt.axhline(y = 1.0, color = 'black', linestyle = '--')
	#plt.legend()
	plt.xlabel("Time bin (100ns)", fontsize=14)
	plt.ylabel("$\\frac{Real}{Sim}$", fontsize=14)
	plt.title("Ratio of Average Pulse Heights", fontsize=14)
	plt.tight_layout()
	plt.savefig("ph_ratio_{}.png".format(fname))
	plt.show()



   