#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 17:19:33 2021

@author: alfonso

This file contains the routines analysing the observables for the different
outbreaks.

"""

import numpy as np
from data_utils import compute_median_age


def measure_first_outbreak(results, mod_id):
    """Measure some observables for the first epidemic outbreak in a given 
    state.
    
    Parameters
    ----------
    results : dict
    
    Contains the following keys:
        path: where data is stored
        country: country where the epidemic unfolds (always United_States)
        state: sub-country regional division
        pop_age: state population distribution by age
        tot_pop: total state population
        deaths: death fraction distribution by age
        mod_id: epidemic model identifier
            time: time array of the epidemic evolution
            pop: population array by age and health-status

    Results
    -------
    output : dict
        observables at the initial state and the steady-state during the first
        outbreak of the epidemic for a given state

    """

    output = {}

    # GLOBAL RESULTS (all age-classes involved)
    
    # Median age
    pop_a = results['pop_age']
    med_age = compute_median_age(pop_a)
    output['mea'] = med_age
    print('Median age {0}'.format(med_age))
    
    # Largest eigenvalue
    output['lam'] = results['lambda']

    # Extract the end of the 1st outbreak
    i_end = results[mod_id]['1st']['i_end']

    # Peak incidence
    inf_tseries = np.sum(results[mod_id]['1st']['pop'][:,:,1], axis=1)
    peak_incidence = np.max(inf_tseries)
    output['pin'] = peak_incidence

    # Time to peak
    i_max = np.argmax(inf_tseries)
    peak_time = results[mod_id]['1st']['time'][i_max]
    output['tpi'] = peak_time
    
    # Susceptible at peak
    sus_pi = np.sum(results[mod_id]['1st']['pop'][i_max].T[0])
    output['spi'] = sus_pi
    
    # Immunized at peak
    imm_pi = 1.0 - sus_pi - peak_incidence
    output['imp'] = imm_pi

    # Initial susceptible
    sus0 = np.sum(results[mod_id]['1st']['pop'][0].T[0])
    output['sus0'] = sus0
        
    # Initial infected
    inf0 = np.sum(results[mod_id]['1st']['pop'][0].T[1])
    output['inf0'] = inf0
        
    # Initial naturally immunized (seroprevalence)
    rem0 = np.sum(results[mod_id]['1st']['pop'][0].T[2])
    output['rem0'] = rem0
    
    # Steady-state susceptible
    sus_ss = np.sum(results[mod_id]['1st']['pop'][i_end].T[0])
    output['sus_ss'] = sus_ss 
        
    # Steady-state infected
    inf_ss = np.sum(results[mod_id]['1st']['pop'][i_end].T[1])
    output['inf_ss'] = inf_ss

    # Steady-state naturally immunized (AND NOT VACCINATED)
    rem_ss = np.sum(results[mod_id]['1st']['pop'][i_end].T[2])
    output['rem_ss'] = rem_ss

    # Herd immunity
    him = 1.0 - 1.0 / 1.5
    output['him'] = him

    # Overshoot
    ove = sus_ss + him - 1.0
    output['ove'] = ove * 100.0

    if (mod_id == 'age_sir_vac' or mod_id == 'age_sir_vac_thr'):

        # Initial vaccinated // and not
        vac0 = np.sum(results[mod_id]['1st']['pop'][0].T[3])
        output['vac0'] = vac0
        nva0 = 1.0 - vac0
        output['nva0'] = nva0

        # Initial total immunized
        imm0 = vac0 + rem0
        output['imm0'] = imm0

        # Steady-state vaccinated // and not
        vac_ss = np.sum(results[mod_id]['1st']['pop'][i_end].T[3])
        output['vac_ss'] = vac_ss
        nva_ss = 1.0 - vac_ss
        output['nva_ss'] = nva_ss

        # Steady-state total immunized
        imm_ss = rem_ss + vac_ss
        output['imm_ss'] = imm_ss

        # Steady-state prevalence (all the people that got sick)
        pre_ss = np.sum(results[mod_id]['1st']['pop'][i_end].T[4])
        output['pre_ss'] = pre_ss

        pre_ss_reset = pre_ss - rem0 # discounting initial prevalence
        output['pre_ss_reset'] = pre_ss_reset

        # Steady-state prevalence by age
        pre_ss_a = results[mod_id]['1st']['pop'][i_end].T[4]
        output['pre_ss_a'] = pre_ss_a
        
        pre_ss_a_reset = pre_ss_a - results[mod_id]['1st']['pop'][0].T[4]
        output['pre_ss_a_reset'] = pre_ss_a_reset

        # Estimated deaths by age
        dea_a = results['deaths'][0] * pre_ss_a
        output['dea_a'] = dea_a
        
        dea_a_reset = results['deaths'][0] * pre_ss_a_reset
        output['dea_a_reset'] = dea_a_reset

        # Total deaths
        dea = np.sum(dea_a)
        output['dea'] = dea
        
        dea_reset = np.sum(dea_a_reset)
        output['dea_reset'] = dea_reset

    return output


def measure_second_outbreak(results, mod_id):
    """Measure some observables for the second epidemic outbreak in a given 
    state.
    
    Parameters
    ----------
    results : dict
    
    Contains the following keys:
        path: where data is stored
        country: country where the epidemic unfolds (always United_States)
        state: sub-country regional division
        pop_age: state population distribution by age
        tot_pop: total state population
        deaths: death fraction distribution by age
        mod_id: epidemic model identifier
            time: time array of the epidemic evolution
            pop: population array by age and health-status

    Results
    -------
    output : dict
        observables at the initial state and the steady-state during the
        second outbreak (isolated from the first) of the epidemic for a given 
        state

    """

    output = {}

    # GLOBAL RESULTS (all age-classes involved)
    
    # Median age
    pop_a = results['pop_age']
    med_age = compute_median_age(pop_a)
    output['mea'] = med_age
    
    # Largest eigenvalue
    output['lam'] = results['lambda']

    # Peak incidence
    inf_tseries = np.sum(results[mod_id]['2nd']['pop'][:,:,1], axis=1)
    peak_incidence = np.max(inf_tseries)
    output['pin'] = peak_incidence

    # Time to peak
    i_max = np.argmax(inf_tseries)
    peak_time = results[mod_id]['2nd']['time'][i_max]
    output['tpi'] = peak_time
    
    # Susceptible at peak
    sus_pi = np.sum(results[mod_id]['2nd']['pop'][i_max].T[0])
    output['spi'] = sus_pi
    
    # Immunized at peak
    imm_pi = 1.0 - sus_pi - peak_incidence
    output['imp'] = imm_pi

    # Initial susceptible
    sus0 = np.sum(results[mod_id]['2nd']['pop'][0].T[0])
    output['sus0'] = sus0
        
    # Initial infected
    inf0 = np.sum(results[mod_id]['2nd']['pop'][0].T[1])
    output['inf0'] = inf0
        
    # Initial naturally immunized (seroprevalence) (discount 1st outbreak)
    rem0 = np.sum(results[mod_id]['2nd']['pop'][0].T[2]) \
           - np.sum(results[mod_id]['1st']['pop'][-1].T[2])
    output['rem0'] = rem0

    # Steady-state susceptible
    sus_ss = np.sum(results[mod_id]['2nd']['pop'][-1].T[0])
    output['sus_ss'] = sus_ss 
        
    # Steady-state infected
    inf_ss = np.sum(results[mod_id]['2nd']['pop'][-1].T[1])
    output['inf_ss'] = inf_ss

    # Steady-state naturally immunized (AND NOT VACCINATED)
    rem_ss = np.sum(results[mod_id]['2nd']['pop'][-1].T[2]) \
             - np.sum(results[mod_id]['1st']['pop'][-1].T[2])
    output['rem_ss'] = rem_ss
    
    # Herd immunity
    him = 1.0 - 1.0 / 3.0
    output['him'] = him

    # Overshoot
    ove = sus_ss + him - 1.0
    output['ove'] = ove * 100.0

    if (mod_id == 'age_sir_vac' or mod_id == 'age_sir_vac_thr'):

        # Initial vaccinated // and not
        vac0 = np.sum(results[mod_id]['2nd']['pop'][0].T[3]) \
               - np.sum(results[mod_id]['1st']['pop'][-1].T[3])
        output['vac0'] = vac0
        nva0 = 1.0 - vac0
        output['nva0'] = nva0
        
        # Initial total immunized
        imm0 = vac0 + rem0
        output['imm0'] = imm0

        # Steady-state vaccinated // and not
        vac_ss = np.sum(results[mod_id]['2nd']['pop'][-1].T[3]) \
                 - np.sum(results[mod_id]['1st']['pop'][-1].T[3])
        output['vac_ss'] = vac_ss
        nva_ss = 1.0 - vac_ss
        output['nva_ss'] = nva_ss

        # Steady-state total immunized
        imm_ss = rem_ss + vac_ss # - imm0
        output['imm_ss'] = imm_ss
        
        # Steady-state prevalence (all the people that got sick) (discount 1st outbreak)
        pre_ss = np.sum(results[mod_id]['2nd']['pop'][-1].T[4]) \
                 - np.sum(results[mod_id]['1st']['pop'][-1].T[4])
        output['pre_ss'] = pre_ss

        pre_ss_reset = pre_ss - rem0 # discounting initial prevalence
        output['pre_ss_reset'] = pre_ss_reset
        
        # Steady-state prevalence by age
        pre_ss_a = results[mod_id]['2nd']['pop'][-1].T[4] \
                   - results[mod_id]['1st']['pop'][-1].T[4]
        output['pre_ss_a'] = pre_ss_a

        # Estimated deaths by age
        dea_a = results['deaths'][0] * pre_ss_a
        output['dea_a'] = dea_a
        
        # Total deaths
        dea = np.sum(dea_a)
        output['dea'] = dea


    return output


def measure_full_epidemic(results, mod_id):
    """Measure some observables for the full epidemic dynamics in a given 
    state.

    Parameters
    ----------
    results : dict
    
    Contains the following keys:
        path: where data is stored
        country: country where the epidemic unfolds (always United_States)
        state: sub-country regional division
        pop_age: state population distribution by age
        tot_pop: total state population
        deaths: death fraction distribution by age
        mod_id: epidemic model identifier
            time: time array of the epidemic evolution
            pop: population array by age and health-status

    Results
    -------
    output : dict
        observables at the initial state and the steady-state during the
        full epidemic for a given state

    """

    output = {}

    # GLOBAL RESULTS (all age-classes involved)
    
    # Median age
    pop_a = results['pop_age']
    med_age = compute_median_age(pop_a)
    output['mea'] = med_age
    
    # Largest eigenvalue
    output['lam'] = results['lambda']

    # Peak incidence
    inf_tseries = np.sum(results[mod_id]['full']['pop'][:,:,1], axis=1)
    peak_incidence = np.max(inf_tseries)
    output['pin'] = peak_incidence

    # Time to peak
    i_max = np.argmax(inf_tseries)
    peak_time = results[mod_id]['full']['time'][i_max]
    output['tpi'] = peak_time
    
    # Susceptible at peak
    sus_pi = np.sum(results[mod_id]['full']['pop'][i_max].T[0])
    output['spi'] = sus_pi
    
    # Immunized at peak
    imm_pi = 1.0 - sus_pi - peak_incidence
    output['imp'] = imm_pi

    # Initial susceptible
    sus0 = np.sum(results[mod_id]['full']['pop'][0].T[0])
    output['sus0'] = sus0
        
    # Initial infected
    inf0 = np.sum(results[mod_id]['full']['pop'][0].T[1])
    output['inf0'] = inf0
        
    # Initial naturally immunized (seroprevalence)
    rem0 = np.sum(results[mod_id]['full']['pop'][0].T[2])
    output['rem0'] = rem0

    # Steady-state susceptible
    sus_ss = np.sum(results[mod_id]['full']['pop'][-1].T[0])
    output['sus_ss'] = sus_ss 
        
    # Steady-state infected
    inf_ss = np.sum(results[mod_id]['full']['pop'][-1].T[1])
    output['inf_ss'] = inf_ss

    # Steady-state naturally immunized (AND NOT VACCINATED)
    rem_ss = np.sum(results[mod_id]['full']['pop'][-1].T[2])
    output['rem_ss'] = rem_ss

    if (mod_id == 'age_sir_vac' or mod_id == 'age_sir_vac_thr'):

        # Initial vaccinated // and not
        vac0 = np.sum(results[mod_id]['full']['pop'][0].T[3])
        output['vac0'] = vac0
        nva0 = 1.0 - vac0
        output['nva0'] = nva0
        
        # Initial total immunized
        imm0 = vac0 + rem0
        output['imm0'] = imm0

        # Steady-state vaccinated // and not
        vac_ss = np.sum(results[mod_id]['full']['pop'][-1].T[3])
        output['vac_ss'] = vac_ss
        nva_ss = 1.0 - vac_ss
        output['nva_ss'] = nva_ss

        # Steady-state total immunized
        imm_ss = rem_ss + vac_ss
        output['imm_ss'] = imm_ss
        
        # Steady-state prevalence (all the people that got sick)
        pre_ss = np.sum(results[mod_id]['full']['pop'][-1].T[4])
        output['pre_ss'] = pre_ss
        pre_ss_reset = pre_ss - rem0 # discounting initial prevalence
        output['pre_ss_reset'] = pre_ss_reset
        
        # Steady-state prevalence by age
        pre_ss_a = results[mod_id]['full']['pop'][-1].T[4]
        output['pre_ss_a'] = pre_ss_a
        
        pre_ss_a_reset = pre_ss_a - results[mod_id]['full']['pop'][0].T[4]
        output['pre_ss_a_reset'] = pre_ss_a_reset
        
        # Estimated deaths by age
        dea_a = results['deaths'][0] * pre_ss_a
        output['dea_a'] = dea_a
        
        dea_a_reset = results['deaths'][0] * pre_ss_a_reset
        output['dea_a_reset'] = dea_a_reset
        
        # Total deaths
        dea = np.sum(dea_a)
        output['dea'] = dea

        dea_reset = np.sum(dea_a_reset)
        output['dea_reset'] = dea_reset
    
    return output