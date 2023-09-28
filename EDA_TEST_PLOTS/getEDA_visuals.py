# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 23:37:03 2022

@author: Nikhiel
"""

import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

from PH_viz import * 


dfr = pd.read_csv("Clean_realsigs_LHC22o.csv")
dfs = pd.read_csv("Clean_simsigs_50k.csv")



#  Heatmap of PH for sim and real data 

phimg(dfs, name="sim")
phimg(dfr, name="real")


# PH heatmaps per SM 
all_ph_sm(dfs, name="sim")
all_ph_sm(dfr, name="real")



# Avg PH per SM 
all_avg_phsm(dfs, name="sim")
all_avg_phsm(dfr, name="real")


# Dual comparison of Avg PH per SM 
Dual_avgphsm(dfs, dfr)


# Ratio of Avg PH per SM

ratio_sm(dfs, dfr)









