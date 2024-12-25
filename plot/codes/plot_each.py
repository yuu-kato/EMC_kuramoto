'''
Functions to plot figures from single EMC simulation data. 
ver. 24-12-06 original
'''

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from matplotlib.markers import MarkerStyle

### General setting for figures
myblue = [0.2, 0.2, 1.0]
mygray = [0.3, 0.3, 0.3]
panellabelsize = 22
paneltitlesize = 18
panellabelsize2 = 30
labelsize = 22
legendsize = 18
plt.rcParams["font.size"] = 13
# plt.rcParams["font.family"] = "DejaVu Serif"

### Functions
# Extract data
def read_data(filepass):
    # true parameters
    param = pd.read_csv(filepass + "/true_parameters.csv")

    # data points
    t, R= np.loadtxt(filepass + "/data.txt", usecols = [0, 1], unpack=True) 

    # free energy
    b, f= np.loadtxt(filepass + "/free_energy.txt", usecols = [0, 1], unpack=True)
    index = np.argmin(f)

    # sampling
    sample = pd.read_csv(filepass + "/parameter.csv")
    sample_burn = sample[sample['mc_step'] > param['burn_in'][0]]

    # MAP
    MAP = pd.read_csv(filepass + "/MAP.csv")

    return param, t, R, b, f, index, sample_burn, MAP

# Fitting
def R_t(t, K, gamma):
    R_0 = 1.0
    R = np.exp(-gamma * t + 0.5 * K * t) / np.sqrt(1.0/R_0/R_0 + (np.exp(-2.0 * gamma * t + K * t) - 1.0) * K / (- 2 * gamma + K))
    return R

def plot_data(t, R, MAP, param, index, fig, gs, label, ylabel, label_sum, mode):
    gamma_estimated = MAP['%s' % index][0]
    K_estimated = 2 * (gamma_estimated - MAP['%s' % index][1])

    if mode=='OA':
        labelstr = label_sum + r' $\sigma=$%s' % param['sigma'][0]
    elif mode=='kuramoto':
        labelstr = label_sum + r' $N=$%s' % param['M'][0]

    # plot
    ax = fig.add_subplot(gs[0,0])
    ax.plot(t, R_t(t, K_estimated, gamma_estimated), color = 'black', lw = 3, label = 'fitting')
    ax.scatter(t, R, color = mygray, label = 'data', s=25)
    ax.set_xlabel(r'$t$', fontsize = labelsize)
    ax.legend(fontsize = legendsize)
    ax.text(-0.1, 1.14, label, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=panellabelsize)
    ax.text(0.5, 1.28, labelstr, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes, fontsize=panellabelsize2)
    if ylabel==True:
        ax.set_ylabel(r'$R(t)$', fontsize = labelsize)

# Free energy
def plot_free(b, f, index, param, fig, gs, label, ylabel, mode):
    xlim = [b[index]*0.1, min(b[index]*2.5, param['b_max'][0])]
    ylim = [f[index]-10, f[index]+50]

    if mode=='OA':
        sigma_infer = (b[index])**(-0.5)
        titlestr = r'$\hat{\sigma}=$%s, $\hat{l}=$%s' % (round(sigma_infer,4), index+1)
    elif mode=='kuramoto':
        titlestr = r'$\hat{l}=$%s' % (index+1)
                                                        
    ax = fig.add_subplot(gs[0,0])
    ax.plot(b, f, color = myblue, lw=5, marker=MarkerStyle("s"), markersize=15, markeredgecolor=myblue, markeredgewidth = 4, markerfacecolor = 'white')
    if mode=='OA':
        sigma_true = param['sigma'][0]
        ax.vlines(sigma_true**(-2), -2000, 2000, color = 'black', linestyle = 'dashed', lw=3, label= 'true')
    ax.vlines(b[index], -2000, 2000, color = 'red', linestyle = 'dashed', lw=3, label=r'infer $(b_\hat{l})$')
    ax.text(-0.1, 1.14, label, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=panellabelsize)
    ax.legend(fontsize = legendsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Inverse temperature', fontsize = labelsize)
    ax.set_title(titlestr, horizontalalignment='right', verticalalignment='bottom', fontsize=paneltitlesize, x=1.0, y=1.0)
    if ylabel==True:
        ax.set_ylabel('Free energy', fontsize = labelsize)

# Determine ylim for 1d histgram
def ylim_1d(mode):
    if mode=='OA':
        ylim = [0, 210]
    elif mode=='kuramoto':
        ylim = [0, 410]
    return ylim

# Histgrams
def plot_K(sample_burn, index, param, MAP, fig, gs, label, ylabel, mode):
    # preparation 
    sample_K = 2 * (sample_burn['l = %s_gamma' % index] - sample_burn['l = %s_delta_gamma' % index])
    K_true = param['K_true'][0]
    K_estimated = 2 * (MAP['%s' % index][0] - MAP['%s' % index][1])
    bin_list = np.arange(np.min(sample_K), np.max(sample_K) + 0.002*2, 0.002)

    # plot
    ax = fig.add_subplot(gs[0,0])
    ax.hist(sample_K, bins = bin_list, histtype = 'step', color = 'black')
    ax.set_xlabel(r'$K$', fontsize = labelsize)
    ax.axvline(K_true, color = 'blue', lw = 3, linestyle = 'dotted', label = 'true')
    ax.axvline(K_estimated, color = 'red', lw = 3, linestyle = 'dotted', label = r'MAP $(\hat{K})$')
    ax.set_xlim(-0.12, 0.22)
    # ax.set_ylim(ylim_1d(mode))
    ax.ticklabel_format(axis="y", scilimits=(-1, 3), useMathText=True)
    ax.legend(fontsize = legendsize)
    ax.text(-0.1, 1.22, label, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=panellabelsize)
    ax.set_title(r'$\hat{K}=$%s' % round(K_estimated, 4), horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=paneltitlesize, x=1.0, y=1.0)
    if ylabel==True:
        ax.set_ylabel('Number of samples', fontsize = labelsize)

def plot_gamma(sample_burn, index, param, MAP, fig, gs, label, ylabel, mode):
    # preparation 
    sample_gamma = sample_burn['l = %s_gamma' % index] 
    gamma_true = param['gamma_true'][0]
    gamma_estimated = MAP['%s' % index][0]
    bin_list = np.arange(np.min(sample_gamma), np.max(sample_gamma) + 0.001*2, 0.001)

    # plot
    ax = fig.add_subplot(gs[0,0])
    ax.hist(sample_gamma, bins = bin_list, histtype = 'step', color = 'black')
    ax.set_xlabel(r'$\gamma$', fontsize = labelsize)
    ax.axvline(gamma_true, color = 'blue', lw = 3, linestyle = 'dotted', label = 'true')
    ax.axvline(gamma_estimated, color = 'red', lw = 3, linestyle = 'dotted', label = r'MAP $(\hat{\gamma})$')
    ax.set_xlim(0.0, 0.16)
    # ax.set_ylim(ylim_1d(mode))
    ax.ticklabel_format(axis="y", scilimits=(-1, 3), useMathText=True)
    ax.legend(fontsize = legendsize)
    ax.text(-0.1, 1.22, label, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=panellabelsize)
    ax.set_title(r'$\hat{\gamma}=$%s' % round(gamma_estimated, 4), horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=paneltitlesize, x=1.0, y=1.0)
    if ylabel==True:
        ax.set_ylabel('Number of samples', fontsize = labelsize)

def plot_2d(sample_burn, index, param, xlim_range, ylim_range, fig, gs, label, ylabel):
    # preparation 
    sample_K = 2 * (sample_burn['l = %s_gamma' % index] - sample_burn['l = %s_delta_gamma' % index])
    sample_gamma = sample_burn['l = %s_gamma' % index] 
    K_true = param['K_true'][0]
    gamma_true = param['gamma_true'][0]
    x=sample_K.to_list()
    y=sample_gamma.to_list()

    # settings for figure
    min_fe = min(np.min(y),np.min(x))
    max_fe = max(np.max(y),np.max(x))
    bins_range = max(abs(min_fe), abs(max_fe))
    bins_x = np.linspace(-bins_range, bins_range, num=1600)
    bins_y = np.linspace(-bins_range, bins_range, num=1600)

    # main heatmap
    ax_main = fig.add_subplot(gs[1:, :-1])
    ax_main.hist2d(x, y, bins=(bins_x, bins_y), cmap='Greys', norm=LogNorm())
    ax_main.axhline(y=gamma_true, color='gray', linestyle='--', linewidth=1)
    ax_main.axvline(x=K_true, color='gray', linestyle='--', linewidth=1)
    ax_main.set_xlabel(r'$K$', fontsize = labelsize)
    ax_main.set_xlim([xlim_range[0], xlim_range[1]])
    ax_main.set_ylim([ylim_range[0], ylim_range[1]])
    ax_main.get_xaxis().set_tick_params(pad=10)
    if ylabel==True:
        ax_main.set_ylabel(r'$\gamma$', fontsize = labelsize)

    # histgram for K
    ax_histx = fig.add_subplot(gs[0, 0:-1], sharex=ax_main)
    ax_histx.hist(x, bins=bins_x, color=mygray, log=False, histtype = 'bar')
    ax_histx.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False) # tick labels
    ax_histx.tick_params(bottom=True, left=False, right=False, top=False) # ticks
    ax_histx.spines['right'].set_visible(False)
    ax_histx.spines['top'].set_visible(False)
    ax_histx.spines['left'].set_visible(False)
    ax_histx.text(-0., 1., label, horizontalalignment='center', verticalalignment='top', transform=ax_histx.transAxes, fontsize=panellabelsize)

    # histgram for gamma
    ax_histy = fig.add_subplot(gs[1:, -1], sharey=ax_main)
    ax_histy.hist(y, bins=bins_y, orientation='horizontal', color=mygray, log=False, histtype = 'bar')
    ax_histy.set(xlabel=None)
    ax_histy.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False) # tick labels
    ax_histy.tick_params(bottom=False, left=True, right=False, top=False) # ticks
    ax_histy.spines['right'].set_visible(False)
    ax_histy.spines['top'].set_visible(False)
    ax_histy.spines['bottom'].set_visible(False)