# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 08:32:36 2025

@author: jppt
"""

import numpy as np
import pandas as pd
import os
import sys
import time
import matplotlib.pyplot as plt
import glob
from copy import deepcopy

from bayes_drt2.inversion import Inverter
from bayes_drt2 import file_load as fl
import bayes_drt2.plotting as bp

from pathlib import Path

# set plotting params and data directory
# # Get the script directory
script_dir = Path(__file__).parent

# locate root directory
root_dir = script_dir.parents[0]

# experimental data path
datadir = root_dir / 'data' 
# datadir = '../data'

tick_size = 9
label_size = 11

plt.rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams['xtick.labelsize'] = tick_size
plt.rcParams['ytick.labelsize'] = tick_size
plt.rcParams['axes.labelsize'] = label_size
plt.rcParams['legend.fontsize'] = tick_size - 1

"Load transmissive planar data"
# load simulated impedance data with noise
Z_file_tp = os.path.join(datadir,'simulated','Z_BimodalTP-DDT_Orazem_0.25.csv')
Zdf_tp = pd.read_csv(Z_file_tp)

# load true DDT
g_file_tp = os.path.join(datadir,'simulated','gamma_BimodalTP-DDT.csv')
g_tp = pd.read_csv(g_file_tp)

# extract frequency and complex impedance
freq_tp = Zdf_tp['Freq'].values
Z_tp = Zdf_tp['Zreal'].values + 1j*Zdf_tp['Zimag'].values

"Load blocking planar data"
# load simulated impedance data with noise
Z_file_bp = os.path.join(datadir,'simulated','Z_BimodalBP-DDT_Orazem_0.25.csv')
Zdf_bp = pd.read_csv(Z_file_bp)

# load true DDT
g_file_bp = os.path.join(datadir,'simulated','gamma_BimodalBP-DDT.csv')
g_bp = pd.read_csv(g_file_bp)

# extract frequency and complex impedance
freq_bp = Zdf_bp['Freq'].values
Z_bp = Zdf_bp['Zreal'].values + 1j*Zdf_bp['Zimag'].values

# plot transmissive planar data
axes = bp.plot_eis(Zdf_tp,bode_cols=['Zreal','Zimag'],s=15,alpha=0.5)
axes[0].set_xlim(0,axes[0].get_xlim()[1])
axes[0].set_ylim(0,axes[0].get_ylim()[1])

# plot blocking planar data
axes = bp.plot_eis(Zdf_bp,bode_cols=['Zreal','Zimag'],s=15,alpha=0.5)
axes[0].set_xlim(0,axes[0].get_xlim()[1])
axes[0].set_ylim(0,axes[0].get_ylim()[1])

inv_tp = Inverter(distributions={'TP-DDT': # user-defined distribution name
                                         {'kernel':'DDT', # indicates that a DDT-type kernel should be used
                                           'dist_type':'parallel', # indicates that the diffusion paths are in parallel
                                           'symmetry':'planar', # indicates the geometry of the system
                                           'bc':'transmissive', # indicates the boundary condition
                                           'ct':False # indicates no simultaneous charge transfer
                                          }
                                },
                  basis_freq=np.logspace(6,-3,91) # use basis range large enough to capture full DDT
                 )

inv_bp = Inverter(distributions={'BP-DDT': # user-defined distribution name
                                         {'kernel':'DDT', # indicates that a DDT-type kernel should be used
                                           'dist_type':'parallel', # indicates that the diffusion paths are in parallel
                                           'symmetry':'planar', # indicates the geometry of the system
                                           'bc':'blocking', # indicates the boundary condition
                                          }
                                },
                  basis_freq=np.logspace(6,-3,91) # use basis range large enough to capture full DDT
                 )


# fit transmissive planar DDT data
start = time.time()
inv_tp.fit(freq_tp, Z_tp, mode="sample")
elapsed = time.time() - start
print('HMC fit time: {:.2f} s'.format(elapsed))

# make a copy of the Inverter object to run the MAP fit
# without overwriting the HMC fit 
inv_tpm = deepcopy(inv_tp)
start = time.time()
inv_tpm.fit(freq_tp,Z_tp, mode="optimize")
elapsed = time.time() - start
print('MAP fit time: {:.2f} s'.format(elapsed))
