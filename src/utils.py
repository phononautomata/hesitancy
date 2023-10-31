#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 22:48:16 2021

@author: alfonso

This file contains some miscelaneous utility functions. 

"""

import os

import numpy as np
import matplotlib.pyplot as plt


def save_plot(path, filename, extension_list):
    """Save the plot as a file.
    
    Parameters
    ----------
    path : string
        directory path
    filename : string
        name of the file to be saved
    extension_list : string list
        list containing file extensions to save it
        
    Returns
    -------
    nothing
    
    """
    
    if not os.path.exists(path):
        os.makedirs(path)
    for ext in extension_list:
        fullName = os.path.join(path, filename + '.' + ext)
        plt.savefig(fullName)
        
        
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def sir_prevalence(tau):
    """Calculate the steady-state removed/recovered prevalence in the SIR
    model by solving the analytical self-consistent equation.
    
    Parameters
    ----------
    tau : np.array
        control parameter array
    
    Returns
    -------
    r_inf : np.array
        steady-state prevalence
    
    """

    # Initialize r_inf
    r_inf = np.zeros(len(tau))

    # Self-consistent solver for r_inf
    for i in range(len(tau)):
        
        guess = 0.8
        escape = 0
        condition = True
        while condition:
            
            r_inf[i] = 1.0 - np.exp(-(tau[i] * guess))
            
            if r_inf[i] == guess:
                condition = False
            
            guess = r_inf[i]
            escape += 1
            
            if escape > 10000:
                r_inf[i] = 0.0
                condition = False
        
    return r_inf


def age_sir_prevalence(mod_id, results, mpars):
    """Calculate the steady-state removed/recovered prevalence in ana ge SIR 
    model by solving the analytical self-consistent equation.
    
    Parameters
    ----------
    -
    
    
    Returns
    -------
    -
    
    """

    beta = mpars['beta']
    gamma = mpars['gamma']

    sus0_a = results[mod_id]['full']['pop'][0].T[0]
    sus_ss_a = results[mod_id]['full']['pop'][-1].T[0]
    rem0_a = results[mod_id]['full']['pop'][0].T[2]
    rem_ss_a = results[mod_id]['full']['pop'][-1].T[2]
    contact = mpars['contact']
    chi_a = mpars['chi_a']
    pop_a = mpars['pop_age'] 
    beta = mpars['beta']
    gamma = mpars['gamma']

    N = np.sum(pop_a)
    cum_inf_a = (pop_a / N) - rem0_a - sus_ss_a
    arg = np.zeros(len(pop_a), dtype=float)
    arg = -beta * N * (contact.dot(cum_inf_a / pop_a)) * chi_a / gamma
    rhs_a = np.exp(arg)
    lhs_a = (sus_ss_a / sus0_a)

    print(np.abs(lhs_a - rhs_a))
    
    rem_ss_ana_a = (pop_a / N) - rhs_a * ((pop_a / N) - rem0_a)

    return np.sum(rem_ss_ana_a)
