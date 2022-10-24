import numpy as np
from scipy.optimize import curve_fit

def power_law(x, c, a):
    '''Decreasing power law: c * x^(-a)'''
    return c*(1/x)**a

def power_law_fit(x,y):
    '''Function to fit a power law and return c(prefactor) and a(decrease exponent)'''

    popt, pcov = curve_fit(power_law,x,y)
    return popt