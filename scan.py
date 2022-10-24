from Simulation import *

#%%
def scan(which_parameter, parameter_list, L = None, xi = None, beta = None, s_center_norm = False):
    
    '''Returns a LIST of simulations over the chosen parameter.'''
    
    simulations = list()
    
    if(which_parameter == 'L'):
        for L in parameter_list:
            sim = Simulation(L, xi , beta)
            sim.generate_fields(s_center_norm = s_center_norm)
            simulations.append(sim)
    elif(which_parameter == 'xi'):
        for xi in parameter_list:
            sim = Simulation(L, xi , beta)
            sim.generate_fields(s_center_norm = s_center_norm)
            simulations.append(sim)
    elif(which_parameter == 'beta'):
        for beta in parameter_list:
            sim = Simulation(L, xi , beta)
            sim.generate_fields(s_center_norm = s_center_norm)
            simulations.append(sim)
    else:
        print(f'{which_parameter} is not a valid parameter.')

    return simulations