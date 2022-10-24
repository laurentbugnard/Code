import numpy as np
from scipy.optimize import curve_fit


def power_law(x, c, a):
    return c*(1/x)**a

def power_law_fit(x,y):
    popt, pcov = curve_fit(power_law,x,y)
    return popt