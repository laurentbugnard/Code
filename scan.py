from Simulation import *

#%%
def scan(L_list, xi_list, beta_list, s_centered = True, vary_seed = False):
    
    '''Returns a LIST of simulations over the chosen parameter.'''
    
    # make a list of seeds with which to simulate
    n = len(L_list) * len(xi_list) * len(beta_list) #total number of simulations
    if(vary_seed):
        seed_list = np.round(np.random.uniform(0,1000, n)) #pick n random seeds
    else:
        seed_list = np.ones(n)
        

    i = 0 #initialize iterator for seeds
    simulations = list() #initialize output
    for L in L_list:
        for xi in xi_list:
            for beta in beta_list:
                sim = Simulation(L, xi , beta)
                sim.generate_fields(s_centered = s_centered, seed = seed_list[i])
                simulations.append(sim)
                i = i+1 #update iterator for seeds
                
    return simulations