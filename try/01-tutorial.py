# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 08:32:36 2025

@author: jppt
"""

import numpy as np
import pandas as pd
import os
import sys
sys.path.append('..')
import time
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# %matplotlib notebook
import glob

from bayes_drt2.inversion import Inverter
import bayes_drt2.file_load as fl
import bayes_drt2.plotting as bp

from pathlib import Path

# %load_ext autoreload
# %autoreload 2

"""
NOTE: the first time you import bayes_drt.inversion, several models will be compiled, which will take a significant
amount of time (typically ~20 minutes). Once compiled, the model files will be stored with the package,
such that this step will only be necessary the first time you import the package.
"""

# Set plot formatting and data directory
# datadir = '../data'
# Get the script directory
script_dir = Path(__file__).parent

# locate root directory
root_dir = script_dir.parents[0]

# experimental data path
datadir = root_dir / 'data' 

tick_size = 9
label_size = 11

plt.rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams['xtick.labelsize'] = tick_size
plt.rcParams['ytick.labelsize'] = tick_size
plt.rcParams['axes.labelsize'] = label_size
plt.rcParams['legend.fontsize'] = tick_size - 1

"Load data"
# load simulated impedance data with noise
Z_file = os.path.join(root_dir,'data','simulated','Z_RC-ZARC_Macdonald_0.25.csv')
Zdf = pd.read_csv(Z_file)

# load true DRT
g_file = os.path.join(datadir,'simulated','gamma_RC-ZARC.csv')
g_true = pd.read_csv(g_file)

# extract frequency and complex impedance
freq, Z = fl.get_fZ(Zdf)

# Plot the data
axes = bp.plot_eis(Zdf)

"Fit the data"
# By default, the Inverter class is configured to fit the DRT (rather than the DDT)
# Create separate Inverter instances for HMC and MAP fits
# Set the basis frequencies equal to the measurement frequencies 
# (not necessary in general, but yields faster results here - see Tutorial 1 for more info on basis_freq)
inv_hmc = Inverter(basis_freq=freq)
inv_map = Inverter(basis_freq=freq)

# Perform HMC fit
start = time.time()
inv_hmc.fit(freq, Z, mode='sample')
elapsed = time.time() - start
print('HMC fit time {:.1f} s'.format(elapsed))

# Perform MAP fit
start = time.time()
inv_map.fit(freq, Z, mode='optimize')  # initialize from ridge solution
elapsed = time.time() - start
print('MAP fit time {:.1f} s'.format(elapsed))

"Visualize DRT and impedance fit"
# plot impedance fit and recovered DRT
fig,axes = plt.subplots(1, 2, figsize=(8, 3.5))

# plot fits of impedance data
inv_hmc.plot_fit(axes=axes[0], plot_type='nyquist', color='k', label='HMC fit', data_label='Data')
inv_map.plot_fit(axes=axes[0], plot_type='nyquist', color='r', label='MAP fit', plot_data=False)

# plot true DRT
p = axes[1].plot(g_true['tau'],g_true['gamma'],label='True',ls='--')
# add Dirac delta function for RC element
axes[1].plot([np.exp(-2),np.exp(-2)],[0,10],ls='--',c=p[0].get_color(),lw=1)

# Plot recovered DRT at given tau values
tau_plot = g_true['tau'].values
# inv_hmc.plot_distribution(ax=axes[1], tau_plot=tau_plot, color='k', label='HMC mean', ci_label='HMC 95% CI')
# inv_map.plot_distribution(ax=axes[1], tau_plot=tau_plot, color='r', label='MAP')

# axes[1].set_ylim(0,3.5)
# axes[1].legend()


# fig.tight_layout()

