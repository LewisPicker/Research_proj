import mesa_reader as mr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb


solar_mass = [5,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
rsol = 6.96e10 #cm
msol = 1.989e33 #grams
lsol = 3.839e33 #ergs
G = 6.67259e-8 #grav constant CGS
z  = 0.0014 #metalicity #solar 0.014

for j in solar_mass:
    print(j)
    
    h = mr.MesaData('/home/lewis/Documents/Honours_Research/data/LOGS/Recombination/001Z/'+str(j)+'M/history.data') #use the path to your data files
    log_L_BAGB = []
    T_BAGB = []
    log_Lhe = []
    for i in range(len(h.star_age)):
        #this should take a section of the He luminosity for when the star is transitioning from core he to shell he burning
        if (h.center_he4[i] < 0.1) and (h.log_LHe[i] < h.log_LZ[i]) and (h.he_core_mass[i] > 0.01*h.star_mass[0] ):
            log_Lhe.append(h.log_LHe[i])
    #we define the base of the AGB to be the minimum in He luminosity.
    tmp = int(np.where(h.log_LHe == min(log_Lhe))[0][0])
    print(tmp)
    #record the surface luminosity and surface temperature.
    log_L_BAGB.append(h.log_L[tmp])
    T_BAGB.append(h.effective_T[tmp])

    print('T_BAGB = ', h.effective_T[tmp])
    print('L_BAGB = ', h.log_L[tmp])
    print('idx = ',  tmp)
    names = ['T_BAGB', 'log_L_BAGB']
    table = [T_BAGB, log_L_BAGB]

    df = pd.DataFrame(table)
    df = df.T
    df.to_csv('/home/lewis/Documents/Honours_Research/data/csv/Reduced/001Z/'+str(j)+'_BAGB_data.csv', index=True, header=names, sep=',') #save the path to your data files
