import mesa_reader as mr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

######## Use this script for stars that develope a significant convective envelope during the AGB phase (typically for lower metalicity stars) ################

solar_mass = [30]  #Unreliable models 27,29,30

rsol = 6.96e10 #cm
msol = 1.989e33 #grams
lsol = 3.839e33 #ergs
G = 6.67259e-8 #grav constant CGS
z  = 0.00014 #metalicity #solar 0.014


#Slicing the data set from the stars minimum radial extent to its maximum during the AGB phase.
for j in solar_mass:
    print(j)
    data = mr.MesaData('/home/lewis/Documents/Honours_Research/data/LOGS/Recombination/001Z/'+str(j)+'M/history.data') #use the path to your data files
    M_core_max = []
    M_conv_max = []
    log_L_TAMS = []
    R_min = []
    R_max = []
    index = len(data.star_age) #number of models
    #get maximum values for M_conv and M_core
    M_core_max.append(data.he_core_mass[index-1])
    M_core_max.append(0.0)
    M_conv_max.append(data.M_conv_env[index-1]/msol)
    M_conv_max.append(0.0)
    print('M_core_max',M_core_max[0])
    print('M_conv_max',M_conv_max[0])

    #getting the index for stat and end of HG (also get the L_TAMS from that)
    radius = []
    #getting minimum and final radius expansion after H shell burning
    max_c12 = int(np.where(data.center_c12 == max(data.center_c12))[0][0])

    radius = data.radius_cm[max_c12:]
    R_min.append(min(radius)/rsol) #min radius roughly at TAMS
    R_min.append(0.0)
    R_max.append(max(radius)/rsol)#max radius
    R_max.append(0.0)
    index_l = int(np.where(data.radius_cm/rsol == R_min[0])[0][0])
    index_u = int(np.where(data.radius_cm/rsol == R_max[0])[0][0])

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
        radius.append(data.radius_cm[i]/rsol)
        M_inter.append(data.M_inter[i]/msol)
        M_conv_env.append(data.M_conv_env[i]/msol)
        T_eff.append(data.effective_T[i])
        he_core.append(data.he_core_mass[i])
        star_age.append(data.star_age[i])
        log_L.append(data.log_L[i])
        M_core.append(data.he_core_mass[i])
        model_number.append(data.model_number[i])
        Ebind_conv.append(data.Ebind_conv[i])
        Egrav_conv.append(data.Egrav_conv[i])
        Etot_conv.append(data.Etot_conv[i])
        Ehe_conv.append(data.Ehe_conv[i])

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

    #Creating a datafile and saving it.
    names = ['model_number','radius','M_inter','M_conv_env','T_eff','he_core_mass','star_age','log_L','M_core', 'Ebind_conv', 'Egrav_conv', 'Etot_conv', 'Ehe_conv', 'M_core_max','M_conv_max','log_L_TAMS','T_norm','R_norm','R_min','R_max']
    table = [model_number,radius,M_inter,M_conv_env,T_eff,he_core, star_age,log_L,M_core, Ebind_conv, Egrav_conv, Etot_conv, Ehe_conv, M_core_max,M_conv_max, log_L_TAMS,T_norm,R_norm,R_min,R_max]

    df = pd.DataFrame(table)
    df = df.T
    df.to_csv('/home/lewis/Documents/Honours_Research/data/csv/Reduced/001Z/'+str(j)+'M_data.csv', index=True, header=names, sep=',') #save the path to your data files








































    #
    # fig, (ax1, ax2, ax3) = plt.subplots(3)
    # plt.title(str(j)+'$M_{\odot}$')
    # plt.xlabel('Model number')
    # ax1.plot(data.model_number, data.radius_cm/rsol)
    # ax1.set_ylabel('Radius $(R_{\odot})$')
    # ax2.plot(data.model_number, data.center_he4)
    # ax2.set_ylabel('center_he4')
    # ax3.plot(data.model_number,data.center_c12)
    # ax3.set_ylabel('center_c12')
    # # ax4.plot(data.model_number, data.log_Lnuc)
    # # ax4.set_ylabel('log_Lnuc')
    # plt.show()
