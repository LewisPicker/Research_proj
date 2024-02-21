import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb
from scipy.optimize import curve_fit

star_mass = [5,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
Z_dir = ['2Z/','1Z/','01Z/','001Z/']
Z = ['2Z','1Z','01Z','001Z']
E_dir = ['Egrav','Ebind','Ehe','Etot']
metalicity = [0.03,0.0142,0.00142,0.000142]
z = 0.0014 # [0.03, 0.014, 0.0014, 0.00014] #2x 1x 0.1X and 0.01X solar metalicity
G = 6.67e-8  #CGS

rsol = 6.957e10 #CGS
msol = 1.989e33 #CGS

#Ebind, Etot, Ehe, Egrav

def quad(x, a, b ,c):
    return a*x**2 + b*x + c

def linear(xdata, m, b):
    return m*xdata + b

def fix_linear(xdata,b):
    return xdata+b


# plot the curves for a look see

# for i in range(len(Z_dir)):
#     for j in range(len(star_mass)):
#         data = pd.read_csv(r'/home/lewis/Documents/Honours_Research/data/csv/Reduced/'+ Z_dir[i] + str(star_mass[j])+'M_data.csv')
#         l_ambda = -(G*star_mass[j]*msol*data.M_conv_env*msol)/(data.Egrav_conv*data.radius*rsol)
#         M_conv_max = data.M_conv_max[0]
#         plt.plot(data.M_conv_env/M_conv_max, l_ambda, label = str(star_mass[j]))
#     # plt.legend()
#     plt.xlabel('Relative Convective mass')
#     plt.ylabel('$\lambda_{Ehe}$')
#     plt.title(Z_dir[i])
#     # plt.ylim(0.8,3.5)
#     # plt.ylim(0.8,3)
#     plt.savefig('/home/lewis/Documents/Honours_Research/data/plots/Lambda_fits/Egrav/' + Z[i] + '.pdf' , format = 'pdf')
#     plt.show()
# exit()



# ############# Uncomment to get the fits to the lambda curves and save the CSV files ###################



# gradient = []
# intercept_01Z = []
# intercept_001Z = []
# for i in range(len(Z_dir)):
#     print('i = ', i)
#     if i < 2:
#         data = pd.read_csv(r'/home/lewis/Documents/Honours_Research/data/csv/Reduced/'+ Z_dir[i] +'15M_data.csv') #fit too 15Msol
#         log_lambda = np.log(-(G*15*msol*data.M_conv_env*msol)/(data.Egrav_conv*data.radius*rsol))
#         M_conv_max = data.M_conv_max[0]
#         MaskMconv= [data.M_conv_env[i] for i in range(len(data.M_conv_env)) if (0.8 > data.M_conv_env[i]/M_conv_max > 0.3)]
#         mask_lambda = [log_lambda[i] for i in range(len(data.M_conv_env)) if (0.8 > data.M_conv_env[i]/M_conv_max > 0.3)]
#         popt, pcov = curve_fit(linear, MaskMconv/M_conv_max , mask_lambda)
#         gradient.append(popt[0])
#         print("The paramaters to the linear lambda fit are:")
#         print("m =", popt[0], "+/-", pcov[0,0]**0.5)
#         print("b =", popt[1], "+/-", pcov[1,1]**0.5)
#         plt.plot(data.M_conv_env/M_conv_max, linear(data.M_conv_env/M_conv_max,*popt), label = 'Linear Fit to 15Msol')
#         for j in range(len(star_mass)):
#             data = pd.read_csv(r'/home/lewis/Documents/Honours_Research/data/csv/Reduced/'+ Z_dir[i] + str(star_mass[j]) +'M_data.csv')
#             log_lambda = np.log(-(G*star_mass[j]*msol*data.M_conv_env*msol)/(data.Egrav_conv*data.radius*rsol))
#             M_conv_max = data.M_conv_max[0]
#             plt.plot(data.M_conv_env/M_conv_max, log_lambda)
#         plt.xlabel('Relative convective mass')
#         plt.ylabel('$Log(\lambda)$')
#         plt.legend()
#         plt.show()
#     else:
#         data = pd.read_csv(r'/home/lewis/Documents/Honours_Research/data/csv/Reduced/'+ Z_dir[i] +'15M_data.csv') #fit to 15Msol
#         log_lambda = np.log(-(G*15*msol*data.M_conv_env*msol)/(data.Egrav_conv*data.radius*rsol))
#         M_conv_max = data.M_conv_max[0]
#         MaskMconv= [data.M_conv_env[i] for i in range(len(data.M_conv_env)) if (0.8 > data.M_conv_env[i]/M_conv_max > 0.3)]
#         mask_lambda = [log_lambda[i] for i in range(len(data.M_conv_env)) if (0.8 > data.M_conv_env[i]/M_conv_max > 0.3)]
#         popt, pcov = curve_fit(linear, MaskMconv/M_conv_max, mask_lambda)
#         gradient.append(popt[0])
#         grad = popt[0]
#         print("The paramaters to the linear lambda fit are:")
#         print("m =", popt[0], "+/-", pcov[0,0]**0.5)
#         print("b =", popt[1], "+/-", pcov[1,1]**0.5)
#         # plt.plot(data.M_conv_env/M_conv_max, linear(data.M_conv_env/M_conv_max,*popt), label = 'Linear Fit to 15Msol')
#         for j in range(len(star_mass)):
#             data = pd.read_csv(r'/home/lewis/Documents/Honours_Research/data/csv/Reduced/'+ Z_dir[i] + str(star_mass[j]) +'M_data.csv')
#             log_lambda = np.log(-(G*star_mass[j]*msol*data.M_conv_env*msol)/(data.Egrav_conv*data.radius*rsol))
#             M_conv_max = data.M_conv_max[0]
#             MaskMconv= np.array([data.M_conv_env[i] for i in range(len(data.M_conv_env)) if (0.8 > data.M_conv_env[i]/M_conv_max > 0.3)])
#             mask_lambda = np.array([log_lambda[i] for i in range(len(data.M_conv_env)) if (0.8 > data.M_conv_env[i]/M_conv_max > 0.3)])
#             popt, pcov = curve_fit(fix_linear,  grad*MaskMconv/M_conv_max, mask_lambda)
#             plt.plot(data.M_conv_env/M_conv_max, log_lambda,label = str(star_mass[j]))
#             plt.plot(data.M_conv_env/M_conv_max, fix_linear(grad*data.M_conv_env/M_conv_max,popt[0]), label = 'Linear fit')
#             plt.xlabel('Relative convective mass')
#             plt.ylabel('$Log(\lambda)$')
#             plt.legend()
#             if i == 2:
#                 intercept_01Z.append(popt[0])
#                 plt.savefig('/home/lewis/Documents/Honours_Research/data/plots/Lambda_fits/Egrav/01Z/' +str(star_mass[j])+ 'M.png',format = 'png')
#                 plt.close()
#             elif i == 3:
#                 intercept_001Z.append(popt[0])
#                 plt.savefig('/home/lewis/Documents/Honours_Research/data/plots/Lambda_fits/Egrav/001Z/' +str(star_mass[j])+ 'M.png',format = 'png')
#                 plt.close()
#
# print('intercept_01Z = ', intercept_01Z )
# print('intercept_001Z = ', intercept_001Z )
# print('Gradient = ', gradient)
#
#
# fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
# axs[0].plot(star_mass, intercept_01Z)
# axs[0].set_ylabel('Intercept')
# axs[0].set_title('Intercepts for 01Z')
# axs[0].set_xlabel('Star Mass')
# axs[1].plot(star_mass, intercept_001Z)
# axs[1].set_title('Intercepts for 001Z')
# axs[1].set_xlabel('Star Mass')
# plt.show()
#
# names = ['intercept_01Z', 'intercept_001Z','grad']
# table = [intercept_01Z,intercept_001Z,gradient]
#
# df = pd.DataFrame(table)
# df = df.T
# df.to_csv('/home/lewis/Documents/Honours_Research/data/csv/Lambda/Egrav_fit_param.csv', index=True, header=names, sep=',') #save the path to your data files
# exit()


############# Ignore Egrav for now.... complete fits for Ebind, Etot, Ehe ############## quadratic fit for Ebind and Ehe Etot use constant gradient. For Etot use m = 0.65

# fitting M for Ebind and Ehe
log_metal = np.log10(metalicity)
dom = np.linspace(log_metal[0],log_metal[-1],100)
m_dir = ['Ebind','Ehe']
for i in m_dir:
    data = pd.read_csv(r'/home/lewis/Documents/Honours_Research/data/csv/Lambda/'+ i +'_fit_param.csv')
    grad = data.grad[:4]
    popt, pcov = curve_fit(quad, log_metal, grad)
    if i == 'Ebind':
        m_Ebind_param = popt
        plt.plot(log_metal,grad, label = 'data')
        plt.plot(dom, quad(dom,*popt), label = 'fit')
        plt.xlabel('log10(metalicity)')
        plt.ylabel('grad')
        plt.title(i)
        plt.legend()
        plt.show()
    elif i == 'Ehe':
        m_Ehe_param = popt
        plt.plot(log_metal,grad, label = 'data')
        plt.plot(dom, quad(dom,*popt), label = 'fit')
        plt.xlabel('log10(metalicity)')
        plt.ylabel('grad')
        plt.title(i)
        plt.legend()
        plt.show()

#fitting for the intercepts in Ebind Ehe and Etot.
# C_dir = ['Ebind', 'Ehe', 'Etot']
dom = np.linspace(star_mass[0],star_mass[-1],100)
for i in E_dir:
    data = pd.read_csv(r'/home/lewis/Documents/Honours_Research/data/csv/Lambda/'+ i +'_fit_param.csv')
    intercept_01Z = data.intercept_01Z
    intercept_001Z = data.intercept_001Z
    popt, pcov = curve_fit(quad, star_mass, intercept_01Z)
    popt1, pcov1 = curve_fit(quad, star_mass, intercept_001Z)
    if i == 'Ebind':
        c_Ebind_01Z_param = popt
        c_Ebind_001Z_param = popt1
        plt.plot(star_mass,intercept_01Z, label = '01Z_data')
        plt.plot(star_mass,intercept_001Z, label = '001Z_data')
        plt.plot(dom, quad(dom,*popt), label = '01Z_fit')
        plt.plot(dom, quad(dom,*popt1), label = '001Z_fit')
        plt.xlabel('Star_mass (M$\odot$)')
        plt.ylabel('Intercept')
        plt.title(i)
        plt.legend()
        plt.show()
    elif i == 'Ehe':
        c_Ehe_01Z_param = popt
        c_Ehe_001Z_param = popt1
        plt.plot(star_mass,intercept_01Z, label = '01Z_data')
        plt.plot(star_mass,intercept_001Z, label = '001Z_data')
        plt.plot(dom, quad(dom,*popt), label = '01Z_fit')
        plt.plot(dom, quad(dom,*popt1), label = '001Z_fit')
        plt.xlabel('Star_mass (M$\odot$)')
        plt.ylabel('Intercept')
        plt.title(i)
        plt.legend()
        plt.show()
    elif i == 'Etot':
        c_Etot_01Z_param = popt
        c_Etot_001Z_param = popt1
        plt.plot(star_mass,intercept_01Z, label = '01Z_data')
        plt.plot(star_mass,intercept_001Z, label = '001Z_data')
        plt.plot(dom, quad(dom,*popt), label = '01Z_fit')
        plt.plot(dom, quad(dom,*popt1), label = '001Z_fit')
        plt.xlabel('Star_mass (M$\odot$)')
        plt.ylabel('Intercept')
        plt.title(i)
        plt.legend()
        plt.show()
    if i == 'Egrav':
        c_Egrav_01Z_param = popt
        c_Egrav_001Z_param = popt1
        plt.plot(star_mass,intercept_01Z, label = '01Z_data')
        plt.plot(star_mass,intercept_001Z, label = '001Z_data')
        plt.plot(dom, quad(dom,*popt), label = '01Z_fit')
        plt.plot(dom, quad(dom,*popt1), label = '001Z_fit')
        plt.xlabel('Star_mass (M$\odot$)')
        plt.ylabel('Intercept')
        plt.title(i)
        plt.legend()
        plt.show()


names = ['m_Ebind_param', 'm_Ehe_param','c_Ebind_01Z_param','c_Ebind_001Z_param','c_Ehe_01Z_param','c_Ehe_001Z_param','c_Etot_01Z_param','c_Etot_001Z_param']
table = [m_Ebind_param, m_Ehe_param, c_Ebind_01Z_param, c_Ebind_001Z_param, c_Ehe_01Z_param, c_Ehe_001Z_param, c_Etot_01Z_param, c_Etot_001Z_param]
df = pd.DataFrame(table)
df = df.T
df.to_csv('/home/lewis/Documents/Honours_Research/data/csv/Lambda/grad_intercepts_param.csv', index=True, header=names, sep=',')

exit()
################
average_grad = []
for i in E_dir:
    print(i)
    data = pd.read_csv(r'/home/lewis/Documents/Honours_Research/data/csv/Lambda/'+ i +'_fit_param.csv')
    average_grad.append(np.average(data.grad[:4])) ##apend average grad we will not fit not a big deal imo e
    plt.plot(np.log10(metalicity), data.grad[:4])
    plt.xlabel('Log10(Z)')
    plt.ylabel('Grad')
    plt.title(i)
    plt.savefig('/home/lewis/Documents/Honours_Research/data/plots/Lambda_fits/'+ i + '_grad.png',format = 'png')
    plt.close()
    if i == 'Ebind':
        plt.title(i)
        plt.plot(star_mass,data.intercept_01Z,label = '01Z')
        plt.plot(star_mass,data.intercept_001Z, label = '001Z')
        plt.legend()
        plt.show()
        plt.close()
    elif i == 'Ehe':
        plt.title(i)
        plt.plot(star_mass,data.intercept_01Z,label = '01Z')
        plt.plot(star_mass,data.intercept_001Z, label = '001Z')
        plt.legend()
        plt.show()
        plt.close()
    elif i == 'Etot':
        plt.title(i)
        plt.plot(star_mass,data.intercept_01Z,label = '01Z')
        plt.plot(star_mass,data.intercept_001Z, label = '001Z')
        plt.legend()
        plt.show()
        plt.close()

print(average_grad)



###### will need to fit for parabolic gradient for Egrav,Ebind, and  linear for Ehe  , Etot is constant  0.65 value.
