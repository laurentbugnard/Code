import pandas as pd
import numpy as np

def get_values(simulation_list):
    df = pd.DataFrame(index = np.arange(len(simulation_list)), \
        columns = ["L", "xi", "beta", "std_s", "seed"])
    
    for i, sim in enumerate(simulation_list):
        df.iloc[i]["L"] = sim.L
        df.iloc[i]["xi"] = sim.xi
        df.iloc[i]["beta"] = sim.beta
        df.iloc[i]["std_s"] = np.std(sim.s)
        df.iloc[i]["seed"] = sim.seed
    
    return df