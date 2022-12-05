import numpy as np


def evolution(system, nstep, max_relaxation_steps = 100000):
    sigma = np.empty([nstep])  # average stress
    epsp = np.empty([nstep])  # average plastic strain
    sigma[0] = system.sigmabar
    epsp[0] = np.mean(system.epsp)

    for i in range(1, nstep+1):
        system.eventDrivenStep(max_steps = max_relaxation_steps)
        sigma[i] = system.sigmabar
        epsp[i] = np.mean(system.epsp)

    if(np.sum((np.diff(epsp) < 0)) > 0):
        print('Warning: epsp not monotonic!')
    return sigma,epsp

def find_runtime_error(system, nstep, max_relaxation_steps = 100000):

    for i in range(1, nstep+1):
        try:
            system.eventDrivenStep(max_steps = max_relaxation_steps)
        except:
            print(f'RuntimeError at step {i}')
            return i
    print('No error found')