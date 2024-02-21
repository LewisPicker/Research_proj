import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb
from scipy.optimize import curve_fit

#####  This script defines and plots the result of the Convective envelope mass function #####

star_mass = [8,12,16,20,24]

z = 0.014 # [0.03, 0.014, 0.0014, 0.00014] #2x 1x 0.1X and 0.01X solar metalicity

gamma = 0.1


b1 = 14.38*(np.log10(z)**2) + 57.41*np.log10(z) + 95.68

a2 = -16.91*(np.log10(z)**2) -81.93*np.log10(z) - 47.87

b2 = 184.02*(np.log10(z)**2) + 872.19*np.log10(z) + 369.75

c2 = -660.1*(np.log10(z)**2) - 3482.35*np.log10(z) + 1488.91


def M_conv_max_func(star_mass, M_core_max,gamma):
    return star_mass - (1+gamma)*M_core_max

def T_norm_func(logLTAMS, a2, b2 ,c2):
    return a2*logLTAMS**2 + b2*logLTAMS + c2

def M_conv_func(T_eff,T_norm,b1,M_conv_max): #removed a since that should be 1 more or less
    return M_conv_max/(1+np.exp(b1*((T_eff/T_norm)-1)))




for i in range(len(star_mass)):
    data = pd.read_csv(r'/home/lewis/Documents/Honours_Research/data/csv/HG_data/1Zsun/'+str(star_mass[i])+'M_data.csv') # Use the path to your data
    M_core_max = data.M_core_max[0]
    M_conv_max = data.M_conv_max[0]
    log_L_TAMS = data.log_L_TAMS[0]

    M_conv_max = M_conv_max_func(star_mass[i], M_core_max,gamma)
    T_norm = T_norm_func(log_L_TAMS, a2, b2 ,c2)
    M_conv = M_conv_func(data.T_eff,T_norm,b1,M_conv_max)
    plt.plot(np.log10(data.T_eff),data.M_conv_env, label = "data")
    plt.plot(np.log10(data.T_eff), M_conv, label = "functions")
    plt.legend()
    plt.show()
