import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

##### this script fits a quadratic to the paramaters a2, b2, c2 and b1 in order to generalise our convetive mass function to varying metalicities ####

#put your metalicities in this array
metalicity = np.log10(np.array([0.03, 0.014, 0.0014, 0.00014]))

# paramaters for the logistical function when fitted to a 15msol curve
b1 = np.array([42.71,36.98,49.82,87.78])
err_b1 = np.array([0.79,0.46,0.79,0.86])

# paramaters of luminocity fit
a2 = np.array([37.86,45.63,48.37,16.71])
err_a2 = np.array([3.37,2.06,5.04,6.33])
b2 = np.array([-529.69,-617.83, -619.01,-258.81])
err_b2 = np.array([28.91,17.55, 44.40, 54.25])
c2 = np.array([5246.76,5698.56,6039.69, 5108.62])
err_c2 = np.array([61.29,36.89,96.66, 115.09])

def linear(x_data,m ,c):
    return m*x_data + c

def quad(x_data, a,b,c):
    return a*x_data**2 + b*x_data + c

#fitting quadratics to the paramater curves
popt1, pcov1 = curve_fit(quad, metalicity, b1)

popt2, pcov2 = curve_fit(quad, metalicity, a2)

popt3, pcov3 = curve_fit(quad, metalicity, b2)

popt4, pcov4 = curve_fit(quad, metalicity, c2)

#print or record the results
print("The paramaters to the linear paramater fits are:")
print("b1_a =", popt1[0], "+/-", pcov1[0,0]**0.5)
print("b1_b =", popt1[1], "+/-", pcov1[1,1]**0.5)
print("b1_c =", popt1[2], "+/-", pcov1[2,2]**0.5)

print("a2_a =", popt2[0], "+/-", pcov2[0,0]**0.5)
print("a2_b =", popt2[1], "+/-", pcov2[1,1]**0.5)
print("a2_c =", popt2[2], "+/-", pcov2[2,2]**0.5)

print("b2_a =", popt3[0], "+/-", pcov3[0,0]**0.5)
print("b2_b =", popt3[1], "+/-", pcov3[1,1]**0.5)
print("b2_c =", popt3[2], "+/-", pcov3[2,2]**0.5)

print("c2_a =", popt4[0], "+/-", pcov4[0,0]**0.5)
print("c2_b =", popt4[1], "+/-", pcov4[1,1]**0.5)
print("c2_c =", popt4[2], "+/-", pcov4[2,2]**0.5)

### Compare the result to the data 
dom = np.linspace(metalicity[0],metalicity[-1],100)

plt.xlabel('Log10(Z)')
plt.ylabel("Values of B1")
plt.errorbar(metalicity,b1,yerr = err_b1, label ='data')
plt.plot(dom, quad(dom,*popt1), label = 'quadratic fit')
plt.legend()
plt.show()


plt.xlabel('Log10(Z)')
plt.ylabel("Values of A2")
plt.errorbar(metalicity,a2,yerr = err_a2, label ='data')
plt.plot(dom, quad(dom,*popt2), label = 'quadratic fit')
plt.legend()
plt.show()

plt.xlabel('Log10(Z)')
plt.ylabel("Values of B2")
plt.plot(dom, quad(dom,*popt3), label = 'quadratic fit')
plt.errorbar(metalicity,b2,yerr = err_b2,label ='data')
plt.legend()
plt.show()

plt.xlabel('Log10(Z)')
plt.ylabel("Values of C2")
plt.plot(dom, quad(dom,*popt4), label = 'quadratic fit')
plt.errorbar(metalicity,c2,yerr = err_c2,label ='data')
plt.legend()
plt.show()
