import pickle

import numpy as np
import matplotlib.pyplot as plt


design_loads = [
    50e3,
    100e3,
    200e3,
    500e3,
    1000e3,
]

for design_load in design_loads:
    with open('GA_%04d_kN_individuals.pickle' % int(design_load*0.001), 'rb') as f:
        individuals = pickle.load(f)
    pop_size = 50
    objectives = [d['outputs']['objective'] for d in individuals]
    it = np.arange(len(objectives))
    min_objectives = np.min(np.asarray(objectives).reshape(-1, pop_size), axis=1)
    plt.plot(it, objectives, 'o', alpha=0.1, mfc='none', zorder=1)
    plt.plot(pop_size*np.arange(len(min_objectives)), min_objectives, '-r', lw=2., zorder=2)
    plt.show()
    break

