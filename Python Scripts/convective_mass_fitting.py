import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb
from scipy.optimize import curve_fit

########## This Script is designed to fit a convective mass function to the MESA data#####

#examples of working datasets for varying metalicity.
# for 2z np.array([5,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
# for 1z np.array([5,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])
# for 01z np.array([5,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
# for 001z np.array([5,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26])
star_mass = [5,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30] # list of star mass in the data set in (solar mass)

rsol = 6.96e10 #cm
msol = 1.989e33 #grams
lsol = 3.839e33 #ergs
G = 6.67259e-8 #grav constant CGS
z  = 0.014 #solar metalicity

Z_dir = '1Z/' #The directory for metalicity specific data set, it minimises the changes you need to make to the scrip when going from one dataset to another.


T_norm_list = []
M_conv_max_list = []
M_core_max_list = []
log_L_TAMS = []
M_inter_f = []
M_inter_min = []
for i in range(len(star_mass)):
    print(star_mass[i])
    data = pd.read_csv(r'/home/lewis/Documents/Honours_Research/data/csv/Reduced/' +Z_dir+str(star_mass[i])+'M_data.csv') # load data
    M_core_max_list.append(data.M_core_max[0])
    M_conv_max_list.append(data.M_conv_max[0])
    T_norm_list.append(data.T_norm[0])
    log_L_TAMS.append(data.log_L_TAMS[0])
    hg_age = np.array(data.star_age)
    M_inter_min.append(min(data.M_inter))
    hg_age = (hg_age - hg_age[0])/(hg_age[-1] - hg_age[0])
    M_inter_f.append(star_mass[i]-M_conv_max_list[i]- M_core_max_list[i])

# Check if the value of gamma converges with this plot!!

gamma_list_f = np.asarray(M_inter_f)/np.asarray(M_core_max_list)
gamma_list_norm = np.asarray(M_inter_min)/np.asarray(M_core_max_list)
plt.plot(star_mass,gamma_list_f, label = 'gamma before core collapse')
plt.plot(star_mass, gamma_list_norm, label = 'gamma at the end of HG')
plt.ylabel('M_inter/M_core_max')
plt.legend()
plt.show()
exit()


log_L_TAMS = np.array(log_L_TAMS)
M_core_max_list = np.array(M_core_max_list)
M_conv_max_list = np.array(M_conv_max_list)


#Use the converging value for gamma typically found that to be around 0.1
gamma = 0.1
#function for M_conv_max
def M_conv_max(star_mass, M_core_max,gamma):
    return star_mass - (1+gamma)*M_core_max

#Function describiing the normalised curves of convective mass vs surface temperature.
def logistic(xdata,b): #removed parameter a == 1
    return 1/(1+np.exp(b*(xdata-1)))

#quadratic function used to fit the L_TAMS to T_norm
def quad(xdata, a, b ,c):
    return a*xdata**2 + b*xdata + c

#linear function used to fit lambda
def linear(xdata, m, b):
    return m*xdata + b

#This plot compares M_conv_max to the function
# plt.plot(star_mass,M_conv_max_list, label = 'data')
# plt.plot(star_mass,M_conv_max(star_mass,M_core_max_list,gamma), label = 'g = 0.1')
# plt.xlabel('Star_mass $(M_{\odot})$')
# plt.ylabel('M_conv_max $(M_{\odot})$')
# plt.legend()
# plt.show()
# exit()

#fitting for T_norm w.r.t TAMS luminocity
popt, pcov = curve_fit(quad, log_L_TAMS, T_norm_list)
print("The paramaters to the luminocity fit are:")
print("a2 =", popt[0], "+/-", pcov[0,0]**0.5)
print("b2 =", popt[1], "+/-", pcov[1,1]**0.5)
print("c2 =", popt[2], "+/-", pcov[2,2]**0.5)
a2 = popt[0]
a2_err = pcov[0,0]**0.5
b2 = popt[1]
b2_err = pcov[1,1]**0.5
c2 = popt[2]
c2_err = pcov[2,2]**0.5
#Check the quality of the above fit with this plot
# plt.plot(log_L_TAMS,T_norm_list, label = 'data')
# plt.plot(log_L_TAMS, quad(log_L_TAMS,*popt), label = 'fit')
# plt.xlabel('log_L_TAMS $(L\odot)$')
# plt.ylabel('T_norm (k)')
# plt.legend()
# plt.title('Quadratic fit to Tnorm')
# plt.show()

#Now to fit the M_conv curve we need to choose a suitible stellar mass curve
#in our case we choose the 15 solar mass the curve you choose to fit too will yeild a slightly more accurate result in that regime
print("fitted to 15Msol")
data = pd.read_csv(r'/home/lewis/Documents/Honours_Research/data/csv/Reduced/' +Z_dir+'15M_data.csv') #load the 15Msol data.
m_index = int(np.where(np.asarray(star_mass) == 15)[0][0]) #locate the chosen mass curve in the array in this case m = 15msol
print('index =', m_index)

#to improve the quality of the fit its best to use a mask to when the convective envelope is changing rapidly
#check what value that is best in thus case we used the threshold data.T_eff[i]/quad(log_L_TAMS[m_index],*popt)<1.2
# plt.plot(data.T_eff/quad(log_L_TAMS[m_index],*popt),data.M_conv_env/M_conv_max_list[m_index])
# plt.xlim(0.95,1.1)
# plt.show()

# The mask been defined by
index= min([i for i in range(len(data.T_eff)) if data.T_eff[i]/quad(log_L_TAMS[m_index],*popt)<1.2])

#Fitting to the logistcal function will applied mask
popt1, pcov1 = curve_fit(logistic, data.T_eff[index:]/quad(log_L_TAMS[m_index],*popt), data.M_conv_env[index:]/M_conv_max(15.0, M_core_max_list[m_index],gamma))
print("The paramaters to the logistical fit are:")
print("b1 =", popt1[0], "+/-", pcov1[0,0]**0.5)
b1 = popt1[0]
b1_err = pcov1[0,0]**0.5



#Full function for M_conv can now be defined
def M_conv_env(star_mass, T_eff, M_core_max, log_L_TAMS):
    return (M_conv_max(star_mass,M_core_max,gamma)*logistic(T_eff/quad(log_L_TAMS,*popt),*popt1))


#plot the results of the fitted function againts the raw data to compare the quality of the results.
# for i in range(len(star_mass)):
#     data = pd.read_csv(r'/home/lewis/Documents/Honours_Research/data/csv/Reduced/'+Z_dir+str(star_mass[i])+'M_data.csv')
#     plt.plot(np.log10(data.T_eff),data.M_conv_env, label = 'data')
#     plt.plot(np.log10(data.T_eff),M_conv_env(star_mass[i],data.T_eff,M_core_max_list[i],log_L_TAMS[i]), label = 'fits')
#     plt.title('Plot comparing M_conv_env function of '+str(star_mass[i])+'Msol')
#     plt.xlabel('$log10(T_{eff}/K)$')
#     plt.ylabel('M_{conv\_env} $(M_{\odot})$')
#     plt.xlim(right = 3.7)
#     plt.legend()
#     # plt.savefig('/home/lewis/Documents/Honours_Research/data/plots/M_conv_fits/mask/'+Z_dir+'/15Msol_fit/'+str(star_mass[i])+'Msol.pdf',format = 'pdf')
#     plt.show()

names = ['b1','b1_err','a2','a2_err','b2','b2_err','c2','c2_err']
table = [b1,b1_err,a2,a2_err,b2,b2_err,c2,c2_err]

df = pd.DataFrame(table)
df = df.T
df.to_csv('/home/lewis/Documents/Honours_Research/data/csv/M_conv_fits/1Z_fit_param.csv', index=True, header=names, sep=',') #save the path to your data files
exit()

#If you would like to have a metalicity specifc function repeat the process for your other datasets and record the values of a2, b2, c2 and b1.
