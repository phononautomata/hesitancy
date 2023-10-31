#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 13:15:52 2021

@author: alfonso

This file contains all plotting functions, those to produce the figures in the
draft/paper, and those for some testing & visualization assitance.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from mpl_toolkits.basemap import Basemap
from matplotlib.colors import rgb2hex, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Polygon
from matplotlib.colorbar import ColorbarBase

import data_utils as du
import utils as ut


### SOME PLOTTING ROUTINES FOR TESTING PURPOSES ###############################

def plot_contact_matrix(state_list):
    '''Plot contact matrix for a series of US states.'''
    
    country = 'United_States'
    path = '/Users/alfonso/Projects/epivac_paper/results'
    
    for state in state_list:
        
        print(state)
        
        # Extract population age distributions & obtain updated contact matrix
        old_pop_a = du.import_age_distribution(path, country, state)
        new_pop_a = du.import_updated_age_distribution(path, country, state)
        contact = du.import_contact_matrix(path, country, state)
        new_contact = du.update_contact_matrix(contact, old_pop_a, new_pop_a)
        
        fig, ax = plt.subplots()
        cax = ax.matshow(new_contact, cmap=plt.cm.Blues)
        fig.colorbar(cax)
        
        #for i in range(len(new_contact)):
        #    for j in range(len(new_contact)):
        #        c = new_contact[j,i]
        #        ax.text(i, j, str(c), va='center', ha='center')
    
        base_name = 'contact.' + state
        ut.save_plot(path, base_name, ['pdf', 'png'])
        

def plot_dynamics(results, mod_id, path):
    '''Plot epidemic global dynamic trajectories for a model in a particular 
    region.'''

    # Prepare figure
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), 
                            constrained_layout=True)

    # Unpack results
    state = results['state']
    N = results['tot_pop']
    i_end = results[mod_id]['1st']['i_end']
    time = results[mod_id]['full']['time']
    pop = results[mod_id]['full']['pop']
    second_wave = results['2ndwave']
    
    # Prepare results
    s_tseries = np.sum(pop[:,:,0], axis=1) / N
    i_tseries = np.sum(pop[:,:,1], axis=1) / N
    
    if mod_id == 'age_sir':
        ind = 2
    elif (mod_id == 'age_sir_vac' or mod_id == 'age_sir_vac_thr'):
        ind = 4
    
    r_tseries = np.sum(pop[:,:,ind], axis=1) / N

    # Plot densities time series
    axs[0].plot(time, s_tseries, linestyle='solid', color='red')
    axs[1].plot(time, i_tseries, linestyle='solid', color='red')
    axs[2].plot(time, r_tseries, linestyle='solid', color='red')

    # Compute & plot homogeneous DHIT
    him_fo = 1.0 / 1.5
    him_so = 1.0 / 3.0

    axs[0].axhline(y=him_fo, color='green', linestyle='dashed')
    axs[0].axhline(y=him_so, color='green', linestyle='dashed')
    
    # Susceptible fraction at peak incidence during 1st outbreak
    i_max = np.argmax(i_tseries)
    sus_pi = np.sum(pop[i_max].T[0]) / N
    
    axs[0].axhline(y=sus_pi, color='limegreen', linestyle='dashed')
    
    # Susceptible fraction at peak incidence during 2nd outbreak
    if second_wave == True:
        inf_tseries2 = np.sum(pop[i_end:,:,1], axis=1) / N
        i_max2 = np.argmax(inf_tseries2)
        sus_pi2 = np.sum(pop[i_end + i_max2].T[0]) / N

        axs[0].axhline(y=sus_pi2, color='limegreen', linestyle='dashed')

    # Compute & plot steady-state remaining susceptible // immunized
    sus_ss1 = np.sum(pop[i_end][:].T[0])
    sus_ss2 = np.sum(pop[-1][:].T[0])
    
    axs[0].axhline(y=sus_ss1, color='lightcoral', linestyle='dashed')
    axs[0].axhline(y=sus_ss2, color='lightcoral', linestyle='dashed')
    
    rem_v = np.sum(pop[i_end][:].T[2])
    sus_v = 1.0 - rem_v

    sus_v = np.sum(pop[i_end][:].T[0])
    imm_v = 1.0 - sus_v
    
    axs[2].axhline(y=imm_v, color='lightcoral', linestyle='dashed')

    # Compute steady-state prevalences
    pre_ss1 = np.sum(pop[i_end][:].T[2])
    #pre_ss2 = np.sum(pop[-1][:].T[2])

    # Plot steady-state prevalence
    axs[2].axhline(y=pre_ss1, color='darkred', linestyle='dotted')
    #axs[2].axhline(y=pre_ss2, color='darkred', linestyle='dotted')
    
    # Compute & plot effective reproductive number
    ax2 = axs[0].twinx() 
    #R_eff1 = 1.5 * np.sum(pop[:i_end,:,0], axis=1) / N
    #ax2.plot(time[:i_end], R_eff1, color='black', linestyle='dotted')
    #if second_wave == True: 
    #    R_eff2 = 3.0 * np.sum(pop[i_end+1:,:,0], axis=1) / N
    #    ax2.plot(time[i_end+1:], R_eff2, color='black', linestyle='dotted')
    ax2.axhline(y=1.0, color='black', linestyle='dashed', linewidth=0.75)
    
    i_ratio1 = np.zeros(i_end)
    for i in range(i_end):    
        i_ratio1[i] = np.sum(pop[i+1].T[1])/np.sum(pop[i].T[1])
    R_eff1 = np.log(i_ratio1) * 4.5 + 1.0
    ax2.plot(time[:i_end], R_eff1, color='blue', linestyle='dotted')
    if second_wave == True:
        i_end2 = len(time) - i_end - 1002
        i_ratio2 = np.zeros(i_end2)
        for i in range(i_end2-1001):    
            i_ratio2[i] = np.sum(pop[i_end+i+2].T[1])/np.sum(pop[i_end+i+1].T[1])
        R_eff2 = np.log(i_ratio2) * 4.5 + 1.0
        ax2.plot(time[i_end+1:-1001], R_eff2, color='blue', linestyle='dotted')

    # Plotting settings
    #fig.suptitle('Epidemic spreading in {0}'.format(results['new']['state']))
    status_list = ['susceptible', 'infected', 'prevalence']

    for ax, i in zip(axs.flatten(), range(len(axs.flatten()))):
        ax.set_xlabel('time (days)')
        ax.set_ylabel('{0} density'.format(status_list[i]))
        #ax.set_ylim(0, 1.025)
        ax.yaxis.set_tick_params(right='on',which='both', direction='in', 
                                 length=4)
        ax.xaxis.set_tick_params(right='on',which='both', direction='in', 
                                 length=4)
        ax.grid(b=True, which='major', c='w', lw=2, ls='-')
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(True)

    base_name = 'dynamics.' + mod_id + '.' + results['state']
    ut.save_plot(path, base_name, ['pdf', 'png'])
    
    
def plot_phases(results, state, path):
    '''Plot a phase diagram for the standard age-structured SIR model.'''

    # Prepare figure
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)

    # Unpack results
    R0_array, ss_pre_array, ss_pre_sol = results
    
    # Plot densities
    ax.plot(R0_array, ss_pre_array, linestyle='solid', color='red', 
            label='age-structured')
    
    # Homogeneous SIR analytical solution
    r_ana = ut.sir_prevalence(R0_array)  
    ax.plot(R0_array, r_ana, color='blue', alpha=0.5, lw=2, 
            label='hom. analytical')
    
    # Age-SIR analytical solution
    ax.plot(R0_array, ss_pre_sol, color='green', linestyle='dotted',
            label='age-struct. sol.')

    # Plotting settings
    ax.set_xlabel('R0')
    ax.set_ylabel('prevalence density')
    ax.set_ylim(0, 1.025)
    ax.yaxis.set_tick_params(right='on',which='both', direction='in', 
                             length=4)
    ax.xaxis.set_tick_params(right='on',which='both', direction='in', 
                             length=4)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(True)

    base_name = 'phase.' + state
    ut.save_plot(path, base_name, ['pdf', 'png'])


def plot_vaccination_comparison(results, path):
    '''Plot a comparison of vaccination campaigns.'''

    # Prepare figure
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 4), 
                            constrained_layout=True)

    # Unpack results
    results1, results2 = results
    N = results1['tot_pop']
    time1 = results1['age_sir_vac']['full']['time']
    pop1 = results1['age_sir_vac']['full']['pop']
    time2 = results2['age_sir_vac']['full']['time']
    pop2 = results2['age_sir_vac']['full']['pop']
    
    # Plot densities
    axs[0,0].plot(time1, np.sum(pop1[:,:,0], axis=1) / N, 
           linestyle='solid', color='red', label='epivac')
    axs[0,0].plot(time2, np.sum(pop2[:,:,0], axis=1) / N,
           linestyle='solid', color='blue', label='morevac')

    axs[0,1].plot(time1, np.sum(pop1[:,:,1], axis=1) / N, 
           linestyle='solid', color='red', label='epivac')
    axs[0,1].plot(time2, np.sum(pop2[:,:,1], axis=1) / N,
           linestyle='solid', color='blue', label='morevac')
    
    axs[1,0].plot(time1, np.sum(pop1[:,:,3], axis=1) / N, 
           linestyle='solid', color='red', label='epivac')
    axs[1,0].plot(time2, np.sum(pop2[:,:,3], axis=1) / N,
           linestyle='solid', color='blue', label='morevac')
    
    axs[1,1].plot(time1, np.sum(pop1[:,:,4], axis=1) / N, 
           linestyle='solid', color='red', label='epivac')
    axs[1,1].plot(time2, np.sum(pop2[:,:,4], axis=1) / N,
           linestyle='solid', color='blue', label='morevac')
    
    # Plot vaccination levels
    iva1 = np.sum(results1['age_sir_vac']['1st']['pop'][0].T[3]) / N
    sva1 = np.sum(results1['age_sir_vac']['1st']['pop'][-1].T[3]) / N
    iva2 = np.sum(results2['age_sir_vac']['1st']['pop'][0].T[3]) / N
    sva2 = np.sum(results2['age_sir_vac']['1st']['pop'][-1].T[3]) / N
    axs[1,0].axhline(y=iva1, color='red', linestyle='dotted')
    axs[1,0].axhline(y=sva1, color='red', linestyle='dotted')
    axs[1,0].axhline(y=iva2, color='blue', linestyle='dotted')
    axs[1,0].axhline(y=sva2, color='blue', linestyle='dotted')

    # Plotting settings
    #fig.suptitle('Epidemic spreading in {0}'.format(results['new']['state']))
    status_list = ['susceptible', 'infected', 'vaccinated', 'prevalence']

    for ax, i in zip(axs.flatten(), range(len(axs.flatten()))):
        ax.set_xlabel('time (days)')
        ax.set_ylabel('{0} density'.format(status_list[i]))
        #ax.set_ylim(0, 1.025)
        ax.yaxis.set_tick_params(right='on',which='both', direction='in', 
                                 length=4)
        ax.xaxis.set_tick_params(right='on',which='both', direction='in', 
                                 length=4)
        ax.grid(b=True, which='major', c='w', lw=2, ls='-')
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(True)

    base_name = 'comp_vaccination.' + results1['state'] + '.' \
                + str(results2['extra_vac'])
    ut.save_plot(path, base_name, ['pdf', 'png'])


### FIGURES TO BE INCLUDED IN THE PAPER #######################################


def plot_figure_1(results, mod_id, path):
    """Plot figure 1. 
    
    A schematic representation of incidence time-evolution distinguishing the 
    two outbreaks. This observable is to be obtained from the age-SIR model 
    without vaccination."""

    # Prepare figure
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)

    # Unpack results
    time = results[mod_id]['full']['time']
    pop = results[mod_id]['full']['pop']
    N = np.sum(results['pop_age'])
    
    # Plot global I density over time
    ax.plot(time, np.sum(pop[:,:,1], axis=1) / N, linestyle='solid', 
            color='red')
    
    # Vertical line separating outbreak 1 from outbreak 2
    ax.axvline(x=250, color='black', linestyle='dotted')
    
    # Plot text
    ax.text(0.2, 0.75, 'Restrictions in place',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes, fontsize=7)
    ax.text(0.2, 0.7, r'$R_0=1.5$',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes, fontsize=7)
    ax.text(0.65, 0.75, 'Restrictions lifted',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes, fontsize=7)
    ax.text(0.65, 0.7, r'$R_0=3$',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes, fontsize=7)

    # Plotting settings
    #ax.set_title("")
    ax.set_xlabel("time (days)")
    ax.set_ylabel("normalized incidence")
    ax.yaxis.set_tick_params(right='on',which='both', direction='in', 
                              length=4)
    ax.xaxis.set_tick_params(right='on',which='both', direction='in', 
                              length=4)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(True)

    base_name = 'figure1.' + results['state']
    ut.save_plot(path, base_name, ['pdf', 'png'])


def plot_figure_2(results, mod_id, path):
    """Plot figure 2. 
    
    A comparison of the population aging from 2005 to 2019 by looking at the
    prevalence by age. Results are obtained for the age-SIR model with a single
    outbreak.

    """

    # Prepare figure
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)

    # Age-structured SIR in worst-case scenario for 2019 population
    #N = results['new']['tot_pop']
    ss_pop_wc = results['new'][mod_id]['full']['pop'][-1][:].T
    pop_a_2019 = np.sum(results['new'][mod_id]['full']['pop'][0][:,0:3], 
                        axis=1)
    attack_rate_by_age_2019 = (ss_pop_wc[2]) #/ pop_a_2019
    age_array = range(len(pop_a_2019))
    
    ax.plot(age_array, attack_rate_by_age_2019, linestyle='-', linewidth=0.2,
            marker='.', color='royalblue', label='2019')

    # Age-structured SIR in worst-case scenario for 2005 population
    #N = results['old']['tot_pop']
    ss_pop_wc = results['old'][mod_id]['full']['pop'][-1][:].T
    #pop_a_2005 = np.sum(results['old']['worst']['full']['pop'][0][:,0:3], axis=1)
    attack_rate_by_age_2005 = (ss_pop_wc[2]) #/ pop_a_2005

    ax.plot(age_array, attack_rate_by_age_2005, linestyle='-',linewidth=0.5,
            marker='.', color='darkorange', label='2005')

    # Plotting settings
    #ax.set_title('Prevalence by age in {0}'.format(results['state']))
    ax.set_xlabel("age")
    ax.set_ylabel("prevalence")
    #ax[0].grid(b=False, zorder=0)
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(True)

    base_name = 'figure2.' + results['new']['state']
    ut.save_plot(path, base_name, ['pdf', 'png'])


def plot_figure_3(results, path):
    """Plot figure 3. 
    
    A comparison of the epidemic trajectories for two states in the age-SIR
    model with/out vaccination.

    """

    # Prepare figure
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), 
                            constrained_layout=True)

    # Unpack results
    results_vac, results_novac = results
    state1 = list(results_vac.keys())[0]
    state2 = list(results_vac.keys())[1]
    mod_id1 = 'age_sir_vac'
    mod_id2 = 'age_sir'
    N_1 = results_vac[state1]['tot_pop']
    N_2 = results_vac[state2]['tot_pop']
    time_e1 = results_vac[state1][mod_id1]['full']['time']
    pop_e1 = results_vac[state1][mod_id1]['full']['pop'] / N_1
    time_n1 = results_novac[state1][mod_id2]['full']['time']
    pop_n1 = results_novac[state1][mod_id2]['full']['pop'] / N_1
    time_e2 = results_vac[state2][mod_id1]['full']['time']
    pop_e2 = results_vac[state2][mod_id1]['full']['pop'] / N_2
    time_n2 = results_novac[state2][mod_id2]['full']['time']
    pop_n2 = results_novac[state2][mod_id2]['full']['pop'] / N_2
    
    # Plot I, R densities over time for different age-structured models    
    axs[0].plot(time_e1, np.sum(pop_e1[:,:,1], axis=1), 
           linestyle='solid', color='darkred', label='vac. ' + state1)
    axs[0].plot(time_n1, np.sum(pop_n1[:,:,1], axis=1),
           linestyle='dotted', color='darkred', label='no vac. ' + state1)
    axs[0].plot(time_e2, np.sum(pop_e2[:,:,1], axis=1), 
           linestyle='solid', color='darkcyan', label='vac. ' + state2)
    axs[0].plot(time_n2, np.sum(pop_n2[:,:,1], axis=1),
           linestyle='dotted', color='darkcyan', label='no vac. ' + state2)

    axs[1].plot(time_e1, np.sum(pop_e1[:,:,4], axis=1), 
           linestyle='solid', color='darkred', label='vac. ' + state1)
    axs[1].plot(time_n1, np.sum(pop_n1[:,:,2], axis=1),
           linestyle='dotted', color='darkred', label='no vac.' + state1)
    axs[1].plot(time_e2, np.sum(pop_e2[:,:,4], axis=1), 
           linestyle='solid', color='darkcyan', label='vac. ' + state2)
    axs[1].plot(time_n2, np.sum(pop_n2[:,:,2], axis=1),
           linestyle='dotted', color='darkcyan', label='no vac. ' + state2)

    # Compute steady-state prevalences
    i_end1 = results_vac[state1][mod_id1]['1st']['i_end']
    i_end2 = results_vac[state2][mod_id1]['1st']['i_end']
    pre1_ss1 = np.sum(pop_e1[i_end1].T[4])
    pre1_ss2 = np.sum(pop_e2[i_end2].T[4])
    pre2_ss1 = np.sum(pop_e1[-1].T[4])
    pre2_ss2 = np.sum(pop_e2[-1].T[4])
    
    # Plot steady-state prevalence
    axs[1].axhline(y=pre1_ss1, color='black', linestyle='dotted')
    axs[1].axhline(y=pre1_ss2, color='black', linestyle='dotted')
    axs[1].axhline(y=pre2_ss1, color='black', linestyle='dotted')
    axs[1].axhline(y=pre2_ss2, color='black', linestyle='dotted')

    # Plotting settings
    #fig.suptitle('Epidemic spreading in {0}'.format(results['new']['state']))
    status_list = ['infected', 'removed']
    
    for ax, i in zip(axs.flatten(), range(len(axs.flatten()))):
        ax.set_xlabel('time (days)')
        ax.set_ylabel('{0} density'.format(status_list[i]))
        #ax.set_ylim(0, 1.025)
        #ax.yaxis.set_tick_params(right='on',which='both', direction='in', 
        #                         length=4)
        #ax.xaxis.set_tick_params(right='on',which='both', direction='in', 
        #                         length=4)
        ax.grid(True)
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(True)
            

    base_name = 'figure3.' + state1 + '.' + state2
    ut.save_plot(path, base_name, ['pdf', 'png'])


def plot_figure_4(global_results, path):
    """Plot figure 4. 
    
    Scatter plots of prevalence or attack rate during second outbreak against
    non-vaccinated fraction (left) and remaining susceptible fraction (right).

    """
    
    # Prepare figure
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), 
                            constrained_layout=True)

    # Prepare data
    n = len(global_results.keys())
    x = np.zeros(n)
    y = np.zeros(n)

    # Collect data (prevalence vs. non-vaccinated) & plot
    for state, i in zip(global_results.keys(), range(n)):
        
        y[i] = global_results[state]['2nd']['pre_ss'] #* 1.0e+6     
        x[i] = global_results[state]['1st']['sus_ss']

    axs[0].scatter(x, y)

    # Add US code labels on all scatter plot points
    for state, i in zip(global_results.keys(), range(n)):
        axs[0].annotate(du.extract_code_from_state(state), (x[i], y[i]))

    # Perform linear regression
    x = x.reshape((-1, 1))
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    intercept = model.intercept_
    coef = model.coef_
    print(intercept)
    print(coef)
    y_pred = model.predict(x)

    # Plot linear fit
    axs[0].plot(x, y_pred, linestyle='solid', color='black', 
                label='R^2={0:.3g}'.format(r_sq))
    
    # Plotting settings
    #ax.set_title("Attack rate after 2nd outbreak")
    axs[0].set_xlabel("non-vaccinated fraction")
    axs[0].set_ylabel("attack rate")
    
    axs[0].yaxis.set_tick_params(right='on',which='both', direction='in', 
                                 length=4)
    axs[0].xaxis.set_tick_params(right='on',which='both', direction='in', 
                                 length=4)
    axs[0].grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = axs[0].legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        axs[0].spines[spine].set_visible(True)

    # Collect data (prevalence vs. remaining susceptible) & plot    
    for state, i in zip(global_results.keys(), range(n)):
        
        y[i] = global_results[state]['2nd']['dea'] #* 1.0e+6      
        x[i] = global_results[state]['1st']['nva_ss']

    axs[1].scatter(x, y)

    # Add labels on all scatter plots points
    for state, i in zip(global_results.keys(), range(n)):
        axs[1].annotate(du.extract_code_from_state(state), (x[i], y[i]))

    # Perform linear regression
    x = x.reshape((-1, 1))
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    intercept = model.intercept_
    coef = model.coef_
    print(intercept)
    print(coef)
    y_pred = model.predict(x)

    # Plot linear fit
    axs[1].plot(x, y_pred, linestyle='solid', color='black', 
            label='R^2={0:.3g}'.format(r_sq))
    
    # Plotting settings
    #ax.set_title("Attack rate during 2nd outbreak")
    axs[1].set_xlabel("non-vaccinated fraction")
    axs[1].set_ylabel("attack rate")
    
    axs[1].yaxis.set_tick_params(right='on',which='both', direction='in', 
                              length=4)
    axs[1].xaxis.set_tick_params(right='on',which='both', direction='in', 
                              length=4)
    axs[1].grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = axs[1].legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        axs[1].spines[spine].set_visible(True)

    base_name = 'figure4_thr'
    ut.save_plot(path, base_name, ['pdf', 'png'])
    

def plot_figure_5A(global_results, path):
    """Plot figure 5A. 
    
    Map of attack rates during the second outbreak along the US states.
    
    """

    m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49, 
                projection='lcc', lat_1=33, lat_2=45, lon_0=-95)
    ax = plt.gca() 
    fig = plt.gcf()
    shp_info = m.readshapefile('/Users/ademiguel/basemap-1.2.2rel/examples/st99_d00', 
                               'states', drawbounds=True)

    nodata_color = "darkorange"
    colors = {}
    statenames = []
    #patches = []
    
    # Rerrange data
    data_dict = {}
    for state in global_results.keys():
        
        value = global_results[state]['2nd']['pre_ss']
        
        state = state.replace("_", " ")
        
        data_dict[state] = value

    cmap = plt.get_cmap('coolwarm')
    vmin = min(data_dict.values()); vmax = max(data_dict.values())
    norm = Normalize(vmin=vmin, vmax=vmax)
    # color mapper to covert values to colors
    mapper = ScalarMappable(norm=norm, cmap=cmap)

    for shapedict in m.states_info:
        statename = shapedict['NAME']
        if statename in data_dict:
            value = data_dict[statename]
            colors[statename] = mapper.to_rgba(value)
            statenames.append(statename)
        else:
            statenames.append(statename)
            colors[statename] = nodata_color

    for nshape,seg in enumerate(m.states):
        color = rgb2hex(colors[statenames[nshape]]) 
        poly = Polygon(seg,facecolor=color,edgecolor=color)
        #if (colors[statenames[nshape]] == nodata_color):
            #p_no = poly
        ax.add_patch(poly)

    # construct custom colorbar
    cax = fig.add_axes([0.27, 0.1, 0.5, 0.05]) # position
    cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.ax.set_xlabel('Attack rates')
    
    base_name = 'figure5A'
    ut.save_plot(path, base_name, ['pdf', 'png'])
    
    plt.show()
    

def plot_figure_5B(global_results, path):
    """Plot figure 5B. 

    Map of non-vaccinated fraction during the second outbreak along the US 
    states.

    """

    m = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49, 
                projection='lcc', lat_1=33, lat_2=45, lon_0=-95)
    ax = plt.gca() 
    fig = plt.gcf()
    shp_info = m.readshapefile('/Users/ademiguel/basemap-1.2.2rel/examples/st99_d00', 
                               'states', drawbounds=True)

    nodata_color = "darkorange"
    colors = {}
    statenames = []
    #patches = []
    
    # Rerrange data
    data_dict = {}
    for state in global_results.keys():
        
        value = global_results[state]['1st']['nva_ss']
        
        state = state.replace("_", " ")
        
        data_dict[state] = value

    cmap = plt.get_cmap('coolwarm')
    vmin = min(data_dict.values()); vmax = max(data_dict.values())
    norm = Normalize(vmin=vmin, vmax=vmax)
    # color mapper to covert values to colors
    mapper = ScalarMappable(norm=norm, cmap=cmap)

    for shapedict in m.states_info:
        statename = shapedict['NAME']
        if statename in data_dict:
            value = data_dict[statename]
            colors[statename] = mapper.to_rgba(value)
            statenames.append(statename)
        else:
            statenames.append(statename)
            colors[statename] = nodata_color

    for nshape,seg in enumerate(m.states):
        color = rgb2hex(colors[statenames[nshape]]) 
        poly = Polygon(seg,facecolor=color,edgecolor=color)
        #if (colors[statenames[nshape]] == nodata_color):
            #p_no = poly
        ax.add_patch(poly)
   
    # construct custom colorbar
    cax = fig.add_axes([0.27, 0.1, 0.5, 0.05]) # posititon
    cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.ax.set_xlabel('Non-vaccinated fraction')
    
    base_name = 'figure5B'
    ut.save_plot(path, base_name, ['pdf', 'png'])
    
    plt.show()
    

def plot_figure_6(global_results, path):
    """Plot figure 6. 
    
    Scatter plots of prevalence or attack rate during second outbreak against
    non-vaccinated fraction when all states have the same fraction of 
    non-vaccinated individuals with respect to their adult population.
    Median age of state is highlighted.

    """
    
    # Prepare figure
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)

    # Prepare data
    n = len(global_results.keys())
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)

    # Collect data (prevalence vs. non-vaccinated) & plot
    for state, i in zip(global_results.keys(), range(n)):
        
        y[i] = global_results[state]['2nd']['pre_ss'] #* 1.0e+6     
        x[i] = global_results[state]['1st']['lam']
        z[i] = du.obtain_median_age(state)

    ax.scatter(x, y)

    # Add US code labels on all scatter plot points
    for state, i in zip(global_results.keys(), range(n)):
        ax.annotate(du.extract_code_from_state(state), (x[i], y[i]), 
                    fontsize=7)

    # Perform linear regression
    x = x.reshape((-1, 1))
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    intercept = model.intercept_
    coef = model.coef_
    print(intercept)
    print(coef)
    y_pred = model.predict(x)

    # Plot linear fit
    ax.plot(x, y_pred, linestyle='solid', color='black', linewidth=0.75,
                label='R^2={0:.3g}'.format(r_sq))
    
    
    # Plot median-age information as colorbar
    cax = ax.scatter(x, y, c=z)
    fig.colorbar(cax)
    
    # Plotting settings
    #ax.set_title("Attack rate after 2nd outbreak")
    ax.set_xlabel("remaining susceptible fraction")
    ax.set_ylabel("attack rate")
    
    ax.yaxis.set_tick_params(right='on',which='both', direction='in', 
                                 length=4)
    ax.xaxis.set_tick_params(right='on',which='both', direction='in', 
                                 length=4)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(True)

    base_name = 'figure6_thr_sus_zeroprev'
    ut.save_plot(path, base_name, ['pdf', 'png'])


def plot_figure_7(results, path):
    """Plot figure 7. 
    
    A comparison of the epidemic trajectories for two states in the age-SIR
    model with baseline vaccination and an extra fraction of vaccinated.

    """

    # Prepare figure
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), 
                            constrained_layout=True)

    # Unpack results
    results_vac, results_exv = results
    state = list(results_vac.keys())[0]
    mod_id = 'age_sir_vac'
    extra_vac = str(results_exv[state]['extra_vac'])
    N = results_vac[state]['tot_pop']
    time_v = results_vac[state][mod_id]['full']['time']
    pop_v = results_vac[state][mod_id]['full']['pop'] / N
    time_e = results_exv[state][mod_id]['full']['time']
    pop_e = results_exv[state][mod_id]['full']['pop'] / N

    # Plot I, R densities over time for different age-structured models    
    axs[0].plot(time_v, np.sum(pop_v[:,:,0], axis=1), 
           linestyle='solid', color='red', label='vac. ')
    axs[0].plot(time_e, np.sum(pop_e[:,:,0], axis=1),
           linestyle='solid', color='blue', label='+' + extra_vac)
    
    axs[1].plot(time_v, np.sum(pop_v[:,:,1], axis=1), 
           linestyle='solid', color='red', label='vac.')
    axs[1].plot(time_e, np.sum(pop_e[:,:,1], axis=1),
           linestyle='solid', color='blue', label='+' + extra_vac)

    axs[2].plot(time_v, np.sum(pop_v[:,:,4], axis=1), 
           linestyle='solid', color='red', label='vac. ')
    axs[2].plot(time_e, np.sum(pop_e[:,:,4], axis=1),
           linestyle='solid', color='blue', label='+' + extra_vac)

    # Compute & ploy herd immunity & overshoots
    him_fo = 1.0 / 1.5
    him_so = 1.0 / 3.0

    axs[0].axhline(y=him_fo, color='green', linestyle='dashed')
    axs[0].axhline(y=him_so, color='green', linestyle='dashed')
    
    # Compute & plot steady-state remaining susceptible // immunized
    sus_v =  np.sum(results_vac[state][mod_id]['1st']['pop'][-1].T[0])
    rem_v = np.sum(results_vac[state][mod_id]['1st']['pop'][-1].T[2])
    vac_v = np.sum(results_vac[state][mod_id]['1st']['pop'][-1].T[3])
    imm_v = rem_v + vac_v
    sus_v1 = 1.0 - imm_v
    rem_e = np.sum(results_exv[state][mod_id]['1st']['pop'][-1].T[2])
    vac_e = np.sum(results_exv[state][mod_id]['1st']['pop'][-1].T[3])
    imm_e = rem_e + vac_e
    sus_e = 1.0 - imm_e

    axs[0].axhline(y=sus_v, color='lightcoral', linestyle='dashed')
    axs[0].axhline(y=sus_v1, color='lightcoral', linestyle='dashed')
    axs[0].axhline(y=sus_e, color='cornflowerblue', linestyle='dashed')
    #axs[2].axhline(y=imm_v, color='lightcoral', linestyle='dashed')
    #axs[2].axhline(y=imm_e, color='cornflowerblue', linestyle='dashed')

    # Compute steady-state prevalences
    i_end_v = results_vac[state][mod_id]['1st']['i_end']-1
    i_end_e = results_exv[state][mod_id]['1st']['i_end']-1
    pre_v_ss1 = np.sum(pop_v[i_end_v].T[4])
    pre_v_ss2 = np.sum(pop_v[-1].T[4])
    pre_e_ss1 = np.sum(pop_e[i_end_e].T[4])
    pre_e_ss2 = np.sum(pop_e[-1].T[4])

    # Plot steady-state prevalence
    axs[2].axhline(y=pre_v_ss1, color='darkred', linestyle='dotted')
    axs[2].axhline(y=pre_v_ss2, color='darkred', linestyle='dotted')
    axs[2].axhline(y=pre_e_ss1, color='darkblue', linestyle='dotted')
    axs[2].axhline(y=pre_e_ss2, color='darkblue', linestyle='dotted')

    # Analytical solution for the second outbreak
    R0 = 3.0 # Default value for the second outbreak
    sus0_v = np.sum(pop_v[i_end_v].T[0])
    rem0_v = np.sum(pop_v[i_end_v].T[2]) + np.sum(pop_v[i_end_v].T[3])
    rem_ss_v = np.sum(pop_v[-1].T[2]) - np.sum(pop_v[i_end_v].T[2]) \
                + np.sum(pop_v[-1].T[3])
    rhs_v = sus0_v * np.exp(-R0 * (rem_ss_v - rem0_v))
    lhs_v = 1.0 - rem_ss_v
    print('Deviation from hom. SIR an. sol.: {0}'.format(rhs_v - lhs_v))

    sus0_e = np.sum(pop_e[i_end_e].T[0])
    rem0_e = np.sum(pop_e[i_end_e].T[2]) + np.sum(pop_e[i_end_e].T[3])
    rem_ss_e = np.sum(pop_e[-1].T[2]) - np.sum(pop_e[i_end_e].T[2]) \
                + np.sum(pop_e[-1].T[3])
    rhs_e = sus0_e * np.exp(-R0 * (rem_ss_e - rem0_e))
    lhs_e = 1.0 - rem_ss_e
    print('Deviation from hom. SIR an. sol.: {0}'.format(rhs_e - lhs_e))

    # Plotting settings
    #fig.suptitle('Epidemic spreading in {0}'.format(results['new']['state']))
    status_list = ['susceptible', 'infected', 'removed']
    
    for ax, i in zip(axs.flatten(), range(len(axs.flatten()))):
        ax.set_xlabel('time (days)')
        ax.set_ylabel('{0} density'.format(status_list[i]))
        #ax.set_ylim(0, 1.025)
        #ax.yaxis.set_tick_params(right='on',which='both', direction='in', 
        #                         length=4)
        #ax.xaxis.set_tick_params(right='on',which='both', direction='in', 
        #                         length=4)
        ax.grid(True)
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(True)

    base_name = 'figure6.' + state + '.' + extra_vac
    ut.save_plot(path, base_name, ['pdf', 'png'])