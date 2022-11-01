from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def power_law(x, c, a):
    '''Decreasing power law: c * x^(-a)'''
    return c*(1/x)**a

def power_law_fit(x,y):
    '''Function to fit a power law and return c(prefactor) and a(decrease exponent)'''

    popt, pcov = curve_fit(power_law,x,y)
    return popt

def plot_power_law_fit(x,y, label = ""):
    
    popt = power_law_fit(x, y)
    plt.scatter(x, y, label = label)
    plt.plot(x, power_law(x, *popt), color = 'k', label = f'|slope| = {popt[0]}')
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.legend()

    return popt[0]