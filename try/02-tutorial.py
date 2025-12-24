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

# fit blocking planar DDT data
# Use ridge initialization to ensure that both chains converge
start = time.time()
inv_bp.fit(freq_bp, Z_bp, mode="sample", init_from_ridge=True)
elapsed = time.time() - start
print('HMC fit time: {:.2f} s'.format(elapsed))

# make a copy of the Inverter object to run the MAP fit
# without overwriting the HMC fit 
inv_bpm = deepcopy(inv_bp)
start = time.time()
inv_bpm.fit(freq_bp, Z_bp, mode="optimize")
elapsed = time.time() - start
print('MAP fit time: {:.2f} s'.format(elapsed))

# call predict_distribution with distribution name specified
inv_tp.predict_distribution('TP-DDT'), \
inv_bp.predict_distribution('BP-DDT')

# Convenience functions for visualization
def plot_result(inv,Zdf,g_true,plot_ci=False,c='k',label=None,axes=None,plot_data=True):
    "Plot recovered DDT and impedance fit"
    if axes is None:
        fig, axes = plt.subplots(2,2,figsize=(7.5,6))
    else:
        fig = axes.ravel()[0].get_figure()
    freq = Zdf['Freq'].values
    
    if label is None:
        Z_label = 'Fit'
        g_label = 'Recovered'
    else:
        Z_label = label
        g_label = label
    
    # plot noisy data
    if plot_data:
        bp.plot_eis(Zdf,axes=axes.ravel()[:3],alpha=0.4,label='Data',bode_cols=['Zreal','Zimag'],s=20)
    # get fitted impedance from Inverter instance
    Z_pred = inv.predict_Z(freq)
    df_pred = fl.construct_eis_df(freq,Z_pred)
    # plot impedance fit
    bp.plot_eis(df_pred,axes=axes.ravel()[:3],alpha=0.8,label=Z_label,bode_cols=['Zreal','Zimag'],
                     c=c,plot_func='plot')

    # plot the true DDT
    if plot_data:
        p = axes[1,1].plot(g_true['tau'],g_true['gamma'],label='True',ls='--')
        # plot data limits
        axes[1,1].axvline(1/(2*np.pi*Zdf['Freq'].min()),ls=':',c='gray',label='Data limits')
        axes[1,1].axvline(1/(2*np.pi*Zdf['Freq'].max()),ls=':',c='gray')
    # get the recovered DDT
    dist_name = list(inv.distributions.keys())[0]
    g_pred = inv.predict_distribution(dist_name,g_true['tau'])
    # plot the recovered DDT
    if plot_ci:
        # plot mean and 95% credibility interval (CI) if HMC sampling used
        g_lo = inv.predict_distribution(dist_name,g_true['tau'],percentile=2.5)
        g_hi = inv.predict_distribution(dist_name,g_true['tau'],percentile=97.5)
        axes[1,1].plot(g_true['tau'],g_pred,c=c,label='Posterior mean',alpha=0.8)
        axes[1,1].fill_between(g_true['tau'],g_lo,g_hi,color=c,label='95% CI',alpha=0.15)
    else:
        # otherwise plot point estimate only
        axes[1,1].plot(g_true['tau'],g_pred,c=c,label=g_label,alpha=0.8)
        
    # plot zero line
    for ax in axes.ravel():
        ax.axhline(0,c='k',lw=0.5,zorder=-10)
        
    axes[1,1].set_xscale('log')
    axes[1,1].set_xlabel(r'$\tau$ / s')
    axes[1,1].set_ylabel(r'$p\,(\ln{\tau})$')
#     axes[1,1].set_ylim(-4.5,2)

    for ax in axes.ravel():
        ax.legend()

    fig.tight_layout()
    return axes

def plot_resid(inv,Zdf):
    "Plot residuals and recovered error scale"
    fig,axes = plt.subplots(1,2,figsize=(7.5,3),sharex=True)
    freq = Zdf['Freq'].values
    Zc = Zdf['Zreal'].values + 1j*Zdf['Zimag'].values
    
    # plot residuals
    Z_pred = inv.predict_Z(freq)
    df_err = fl.construct_eis_df(freq,Z_pred-Zc)
    bp.plot_bode(df_err,axes=axes,cols=['Zreal','Zimag'],alpha=0.4,s=20,label='Residuals',unit_scale='')

    # plot true error scale
    p = axes[0].plot(freq,3*Zdf['sigma_re'],ls='--',label='True $\pm 3\sigma$')
    axes[0].plot(freq,-3*Zdf['sigma_re'],ls='--',c=p[0].get_color())
    axes[1].plot(freq,3*Zdf['sigma_im'],ls='--')
    axes[1].plot(freq,-3*Zdf['sigma_im'],ls='--',c=p[0].get_color())
    
    # plot zero line
    for ax in axes:
        ax.axhline(0,c='k',lw=0.5,zorder=-10)

    # get the recovered error scale
    sigma_re, sigma_im = inv.predict_sigma(freq)
    # plot recovered error scale
    axes[0].fill_between(freq,-3*sigma_re,3*sigma_re,color='k',alpha=0.15,label='Recovered $\pm3\sigma$')
    axes[1].fill_between(freq,-3*sigma_im,3*sigma_im,color='k',alpha=0.15)

    axes[0].legend()
    
    fig.tight_layout()
    return axes

# plot HMC results for transmissive planar DDT
axes = plot_result(inv_tp,Zdf_tp,g_tp,plot_ci=True,label='HMC')
# overlay the MAP results
axes = plot_result(inv_tpm,Zdf_tp,g_tp,plot_data=False,axes=axes,label='MAP',c='r')

#  plot residuals and error scale for HMC fit
axes1 = plot_resid(inv_tp,Zdf_tp)
fig1 = axes1[0].get_figure()
fig1.suptitle('HMC result')
fig1.subplots_adjust(top=0.88)
#  plot residuals and error scale for MAP fit
axes2 = plot_resid(inv_tpm,Zdf_tp)
fig2 = axes2[0].get_figure()
fig2.suptitle('MAP result')
fig2.subplots_adjust(top=0.88)

print('Ohmic resistance = {:.5f} ohms'.format(inv_tp.R_inf))
print('Inductance = {:.5e} H'.format(inv_tp.inductance))
print('Polarization resistance = {:.5f} ohms'.format(inv_tp.predict_Rp()))
# print('Rp 2.5 percentile = {:.5f} ohms'.format(inv_tp.predict_Rp(percentile=2.5)))
# print('Rp 97.5 percentile = {:.5f} ohms'.format(inv_tp.predict_Rp(percentile=97.5)))

# plot HMC results for blocking planar DDT
axes = plot_result(inv_bp,Zdf_bp,g_bp,plot_ci=True,label='HMC')
# overlay the MAP results
axes = plot_result(inv_bpm,Zdf_bp,g_bp,plot_data=False,axes=axes,label='MAP',c='r')

#  plot residuals and error scale for HMC fit
axes1 = plot_resid(inv_bp,Zdf_bp)
fig1 = axes1[0].get_figure()
fig1.suptitle('HMC result')
fig1.subplots_adjust(top=0.88)
#  plot residuals and error scale for MAP fit
axes2 = plot_resid(inv_bpm,Zdf_bp)
fig2 = axes2[0].get_figure()
fig2.suptitle('MAP result')
fig2.subplots_adjust(top=0.88)

print('Ohmic resistance = {:.5f} ohms'.format(inv_bp.R_inf))
print('Inductance = {:.5e} H'.format(inv_bp.inductance))
print('Polarization resistance = {:.5e} ohms'.format(inv_bp.predict_Rp()))
# print('Rp 2.5 percentile = {:.5e} ohms'.format(inv_bp.predict_Rp(percentile=2.5)))
# print('Rp 97.5 percentile = {:.5e} ohms'.format(inv_bp.predict_Rp(percentile=97.5)))
