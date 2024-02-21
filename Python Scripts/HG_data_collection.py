import mesa_reader as mr
import pandas as pd
import numpy as np
import pdb

######## Use this script for stars that develope a significant convective envelope during the HG phase ################

solar_mass = [5,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
rsol = 6.96e10 #cm
msol = 1.989e33 #grams
lsol = 3.839e33 #ergs
G = 6.67259e-8 #grav constant CGS
z  = 0.0014 #metalicity #solar 0.014



#Slicing the data set from the stars minimum radial extent to its maximum during the HG phase.
for j in solar_mass:
    print('star mass=', j)
    h = mr.MesaData('/home/lewis/Documents/Honours_Research/data/LOGS/Recombination/01Z/'+str(j)+'M/history.data') #use the path to your data files
    M_core_max = []
    M_conv_max = []
    log_L_TAMS = []
    R_min = []
    R_max = []
    M_core_max_hg = []
    M_conv_max_hg = []
    index = len(h.star_age)
    M_core_max.append(h.he_core_mass[index-1])
    M_conv_max.append(h.M_conv_env[index-1]/msol)
    print('M_core_max',M_core_max[0])
    print('M_conv_max',M_conv_max[0])


    radius = []
    for i in range(len(h.star_age)):
        if (h.center_he4[i] > 0.8): #this threshhold for the concentration of he4 it is a reaonably large threshold since peak star expansion occurs after core helium ignition
            radius.append(h.radius_cm[i])

    R_min.append(min(radius)/rsol) #min radius roughly at TAMS
    R_min.append(0.0)
    R_max.append(max(radius)/rsol)#max radius
    R_max.append(0.0)
    index_l = int(np.where(h.radius_cm/rsol == R_min[0])[0][0]) #lower index
    index_u = int(np.where(h.radius_cm/rsol == R_max[0])[0][0]) #upper index
    log_L_TAMS.append(h.log_L[index_l])
    M_conv_max_hg.append(max(h.M_conv_env[index_l:index_u])/msol) #appending maximum convective mass in HG
    M_core_max_hg.append(max(h.he_core_mass[index_l:index_u])) #appending maximum core mass in HG
    print('Minimum radius = ', R_min[0], "Maximum radius = ", R_max[0])
    print('lower index =',index_l,'upper index =',index_u)


    #Appending the paramaters that we are interested in for the project
    radius = []
    M_inter= []
    M_conv_env = []
    T_eff = []
    he_core = []
    star_age = []
    log_L = []
    M_core = []
    model_number = []
    Ebind_conv = []
    Egrav_conv =[]
    Etot_conv = []
    Ehe_conv = []
    for i in list(range(index_l,index_u)):
        radius.append(h.radius_cm[i]/rsol)
        M_inter.append(h.M_inter[i]/msol)
        M_conv_env.append(h.M_conv_env[i]/msol)
        T_eff.append(h.effective_T[i])
        he_core.append(h.he_core_mass[i])
        star_age.append(h.star_age[i])
        log_L.append(h.log_L[i])
        M_core.append(h.he_core_mass[i])
        model_number.append(h.model_number[i])
        Ebind_conv.append(h.Ebind_conv[i])
        Egrav_conv.append(h.Egrav_conv[i])
        Etot_conv.append(h.Etot_conv[i])
        Ehe_conv.append(h.Ehe_conv[i])

    ###Calculating Index for T_norm when M_conv/M_conv_max = 0.5
    T_norm = []
    R_norm = []
    M_conv_array = array = np.asarray(M_conv_env)
    idx = (np.abs(M_conv_array/M_conv_max[0] - 0.5)).argmin()
    T_norm.append(T_eff[idx])
    T_norm.append(0.0)
    R_norm.append(radius[idx])
    R_norm.append(0.0)
    print('T_norm = ', T_norm[0])
    print('R_norm = ', R_norm[0])



    #new variables M_core_max_hg, M_conv_max_hg, T_BAGB, L_BAGB
    #Creating a datafile and saving it.
    names = ['model_number','radius','M_inter','M_conv_env','T_eff','he_core_mass','star_age','log_L','M_core', 'Ebind_conv', 'Egrav_conv', 'Etot_conv', 'Ehe_conv','M_core_max','M_conv_max','log_L_TAMS','T_norm','R_norm','R_min','R_max','M_core_max_hg', 'M_conv_max_hg']
    table = [model_number,radius,M_inter,M_conv_env,T_eff,he_core, star_age,log_L,M_core, Ebind_conv, Egrav_conv, Etot_conv, Ehe_conv, M_core_max,M_conv_max, log_L_TAMS,T_norm,R_norm,R_min,R_max,M_core_max_hg, M_conv_max_hg]

    df = pd.DataFrame(table)
    df = df.T
    df.to_csv('/home/lewis/Documents/Honours_Research/data/csv/Reduced/01Z/'+str(j)+'M_data.csv', index=True, header=names, sep=',') #save the path to your data files
