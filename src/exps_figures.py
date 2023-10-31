#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 20:10:37 2021

@author: alfonso

This file contains the experiments to produce the figures in the draft/paper.

"""

import numpy as np
import  csv

from sklearn.linear_model import LinearRegression

from data_utils import feed_the_model
from utils import age_sir_prevalence
from model_age_sir import call_age_sir
from model_age_sir_vac import call_age_sir_vac
from age_sir_vac_th import call_age_sir_vac_thr
import measure_outbreaks as mo
import plots as pt


path = '/Users/ademiguel/Workshop'
country = 'United_States'

state_list = ['Alaska', 'Alabama', 'Arkansas', 'Arizona', 'California', 
              'Colorado', 'Connecticut', 'District_of_Columbia', 'Delaware', 
              'Florida', 'Georgia', 'Hawaii', 'Iowa', 'Idaho', 'Illinois', 
              'Indiana', 'Kansas', 'Kentucky', 'Louisiana', 'Massachusetts', 
              'Maryland', 'Maine', 'Michigan', 'Minnesota', 'Missouri', 
              'Mississippi', 'Montana', 'North_Carolina', 'North_Dakota', 
              'Nebraska', 'New_Hampshire', 'New_Jersey', 'New_Mexico', 
              'Nevada', 'New_York', 'Ohio', 'Oklahoma', 'Oregon', 
              'Pennsylvania', 'Rhode_Island', 'South_Carolina', 'South_Dakota', 
              'Tennessee', 'Texas', 'Utah', 'Virginia', 'Vermont', 
              'Washington', 'Wisconsin', 'West_Virginia', 'Wyoming']


def produce_phases():
    
    # Obtain data
    state = 'Massachusetts'
    update = True
    zeroprev = True
    mpars = feed_the_model(path, country, state, update, zeroprev)
    
    mpars['seeds'] = mpars['seeds'] / 1000.0
    mpars['R0_P2'] = 0.0
    mpars['second_wave'] = False
    mod_id = 'age_sir'

    R0_array = np.linspace(0.0, 3.0, 100)
    ss_pre_array = np.zeros(len(R0_array))
    ss_pre_sol = np.zeros(len(R0_array))

    # Explore phases
    for R0, i in zip(R0_array, range(len(R0_array))):
        
        print('R0={0}'.format(R0))
        
        mpars['R0'] = R0

        results = call_age_sir(mpars)
        
        ss_pre_sol[i] = age_sir_prevalence(mod_id, results, mpars)
        
        ss_pre_array[i] = np.sum(results[mod_id]['full']['pop'][-1].T[2])

    results = R0_array, ss_pre_array, ss_pre_sol
    pt.plot_phases(results, state, path)


def produce_figure_1():
    """Produce figure 1.
    
    Incidence as a function of time during both outbreaks without vaccination
    for a particular US state. Only restrictions scenario.

    """

    # Obtain data
    state = 'Vermont'
    mpars = feed_the_model(path, country, state)

    # Invoke Runge-Kutta
    results = call_age_sir(mpars)

    # Plot
    mod_id = 'age_sir'
    full_path = path + '/epivac_paper/results'
    pt.plot_figure_1(results, mod_id, full_path)


def produce_figure_2():
    """Produce figure 2. 
    
    A comparison of the prevalence by age class between 2005 and 2019
    populations and contact matrices. It should show the aging of the 
    population overall. Results are obtained by the worst-case scenario model: 
    age-SIR with a single outbreak, no vaccination, no restrictions.

    """

    results = {}

    # Obtain data for model 1 (updated populations)
    state = 'Florida'
    update = True
    mpars = feed_the_model(path, country, state, update)
    
    # Set R0 for the single wave
    mpars['R0'] = 3.0
    mpars['second_wave'] = False

    # Invoke Runge-Kutta & store results for model 1
    results1 = call_age_sir(mpars)
    results['new'] = results1
        
    # Obtain data for model 2 (old populations)
    update = False
    mpars2 = feed_the_model(path, country, state, update)

    # Set R0 for the single wave
    mpars2['R0'] = 3.0
    mpars2['second_wave'] = False
        
    # Invoke Runge-Kutta & store results for model 2
    results2 = call_age_sir(mpars2)
    results['old'] = results2

    # Plot
    mod_id = 'age_sir'
    full_path = path + '/epivac_paper/results'
    pt.plot_figure_2(results, mod_id, full_path)


def produce_figure_3():
    """Produce figure 3. 
    
    A comparison of epidemic trajectories for two states in the age-SIR model
    with and without vaccination.
    
    """

    results_vac = {}
    results_novac = {}

    state1 = 'Oklahoma'
    state2 = 'Vermont'

    # Obtain data for state 1 in model with vaccination
    state = state1
    update = True
    zeroprev = True
    mpars1 = feed_the_model(path, country, state, update, zeroprev)
        
    # Vaccination campaign & 2nd outbreak
    mpars1['vac_id'] = 'pho'
 
    # Invoke Runge-Kutta & store results for state 1 in model with vaccination
    results_v1 = call_age_sir_vac(mpars1)
    results_vac[state] = results_v1
    
    # Obtain data for state 2 in model with vaccination
    state = state2
    mpars2 = feed_the_model(path, country, state, update, zeroprev)
        
    # Vaccination campaign & 2nd outbreak
    mpars2['vac_id'] = 'pho'

    # Invoke Runge-Kutta & store results for state 2 in model with vaccination
    results_v2 = call_age_sir_vac(mpars2)
    results_vac[state] = results_v2
        
    # Obtain data for state 1 in model without vaccination
    state = state1
    mpars1 = feed_the_model(path, country, state, update, zeroprev)

    # Invoke Runge-Kutta & store results for state 1 in model without vacc.
    results_v1 = call_age_sir(mpars1)
    results_novac[state] = results_v1
    
    # Obtain data for state 2 in model without vaccination
    state =  state2
    mpars2 = feed_the_model(path, country, state, update, zeroprev)

    # Invoke Runge-Kutta & store results for state 2 in model without vacc.
    results_n2 = call_age_sir(mpars2)
    results_novac[state] = results_n2

    full_path = path + '/epivac_paper/results'
    total_results = results_vac, results_novac
    pt.plot_figure_3(total_results, full_path)


def produce_figures_4_and_5():
    """ Produce figure 4 and 5 (5A & 5B).

    Since figures 4 and 5 need computations for every state, these are produced 
    in the same run in order to save time. 
    
    Figure 4 contains two scatter plots of attack rate during the second
    outbreak versus the non-vaccinated fraction (left) and the remanining
    susceptible fraction at the beginning og the 2nd outbreak (right).
    
    Figure 5A represents the geographical US map at the state level of attack 
    rates during the 2nd outbreak. Figure 5B represents the geographical US map 
    at the state level of non-vaccinated fractions during the 2nd outbreak.
    Of course, these could be subplots of the same plot but there was some 
    problem and I gave up soon.

    """
    
    # Prepare dictionary for state results
    global_results = {}

    # Loop over states
    for state in state_list:
    
        print('{0}'.format(state))

        # Obtain data
        update= True
        zeroprev = False
        counterfact = False
        mpars = feed_the_model(path, country, state, update, zeroprev, 
                               counterfact)
        
        # Vaccination campaign
        mpars['vac_id'] = 'pho'
        
        # Invoke Runge-Kutta
        mod_id = 'age_sir_vac'
        results = call_age_sir_vac(mpars)
    
        # Perform measurements
        results_1w = mo.measure_first_outbreak(results, mod_id)
        results_2w = mo.measure_second_outbreak(results, mod_id)
        results_full = mo.measure_full_epidemic(results, mod_id)

        # Store into global dictionary
        global_results[state] = {'1st': results_1w, 
                                 '2nd': results_2w,
                                 'full': results_full}

    # Plotting section
    full_path = path + '/epivac_paper/results'
    pt.plot_figure_5A(global_results, full_path)
    pt.plot_figure_5B(global_results, full_path)
    pt.plot_figure_4(global_results, full_path)


def produce_tables():
    """Produce tables with differences for observables in baseline 
    vaccination model and extra vaccination model."""

    # Initialize table 1
    dealist = []
    dealist.append(['state', 'averted deaths (1st wave)', 
                      'averted deaths (2nd wave)', 'averted deaths (full)'])
    
    prelist = []
    prelist.append(['state', 'diff. prev. (1st wave)', 'diff. prev. (2nd wave)',
                     'diff. prev. (full)'])
    
    immlist = []
    immlist.append(['state', 'diff. imm. (1st wave)', 'diff. imm. (2nd wave)',
                     'diff. imm. (full)'])
    
    suslist = []
    suslist.append(['state', 'diff. imm. (1st wave)', 'diff. imm. (2nd wave)',
                     'diff. imm. (full)'])
    
    corlist = []

    # Loop over states
    for state in state_list:

        print('{0}'.format(state))

        # Obtain data 
        update= True # Fed with 2019 census data
        zeroprev = False
        counterfact = True
        mpars = feed_the_model(path, country, state, update, zeroprev, 
                               counterfact)

        # Vaccination campaign
        mpars['vac_id'] = 'thr'
        mpars['extra_vac'] = 0.0

        # Invoke Runge-Kutta for the baseline vaccination model
        mod_id = 'age_sir_vac_thr'
        res_bl = call_age_sir_vac_thr(mpars)

        # Perform experiments
        res_bl_1st = mo.measure_first_outbreak(res_bl, mod_id)
        res_bl_2nd = mo.measure_second_outbreak(res_bl, mod_id)
        res_bl_fe = mo.measure_full_epidemic(res_bl, mod_id)

        # Extract observables
        nva_bl_1st = res_bl_1st['nva_ss']
        dea_bl_1st = res_bl_1st['dea_reset']
        dea_bl_2nd = res_bl_2nd['dea']
        dea_bl_fe = res_bl_fe['dea']
        pre_bl_1st = res_bl_1st['pre_ss_reset']
        pre_bl_2nd = res_bl_2nd['pre_ss']
        pre_bl_fe = res_bl_fe['pre_ss']
        imm_bl_1st = res_bl_1st['imm_ss']
        imm_bl_2nd = res_bl_2nd['imm_ss']
        imm_bl_fe = res_bl_fe['imm_ss']
        sus_bl_1st = res_bl_1st['sus_ss']
        sus_bl_2nd = res_bl_2nd['sus_ss']

        # Informative print
        print('Baseline vaccination model for {0}'.format(state))
        print('Non vaccinated: {0}'.format(nva_bl_1st))
        print('Remaining susceptible: {0}'.format(sus_bl_1st))
        print('Remanining susceptible after 2nd outbreak: {0}'.format(sus_bl_2nd))
        print('Attacked in 1st outbreak: {0}'.format(pre_bl_1st))
        print('Attacked in 2nd outbreak: {0}'.format(pre_bl_2nd))
        print('Attacked in full epidemic: {0}'.format(pre_bl_fe))
        print('Deaths in 1st outbreak:{0}'.format(dea_bl_1st))
        print('Deaths in 2nd outbreak:{0}'.format(dea_bl_2nd))
        print('Deaths in full epidemic:{0}'.format(dea_bl_fe))
        print('Immunized in 1st outbreak: {0}'.format(imm_bl_1st))
        print('Immunized in 2nd outbreak: {0}'.format(imm_bl_2nd))
        print('Immunized in full epidemic: {0}'.format(imm_bl_fe))

        # Vaccination campaign
        mpars['vac_id'] = 'thr'
        mpars['extra_vac'] = 0.01

        # Invoke Runge-Kutta for extra vaccination campaign model
        mod_id = 'age_sir_vac_thr'
        res_ev = call_age_sir_vac_thr(mpars)

        # Perform experiment
        res_ev_1st = mo.measure_first_outbreak(res_ev, mod_id)
        res_ev_2nd = mo.measure_second_outbreak(res_ev, mod_id)
        res_ev_fe = mo.measure_full_epidemic(res_ev, mod_id)

        # Extract observables
        nva_ev_1st = res_ev_1st['nva_ss']
        dea_ev_1st = res_ev_1st['dea_reset']
        dea_ev_2nd = res_ev_2nd['dea']
        dea_ev_fe = res_ev_fe['dea']
        pre_ev_1st = res_ev_1st['pre_ss_reset']
        pre_ev_2nd = res_ev_2nd['pre_ss']
        pre_ev_fe = res_ev_fe['pre_ss']
        imm_ev_1st = res_ev_1st['imm_ss']
        imm_ev_2nd = res_ev_2nd['imm_ss']
        imm_ev_fe = res_ev_fe['imm_ss']
        sus_ev_1st = res_ev_1st['sus_ss']
        sus_ev_2nd = res_ev_2nd['sus_ss']

        print('+1%N extra vaccination model for {0}'.format(state))
        print('Non vaccinated: {0}'.format(nva_ev_1st))
        print('Remaining susceptible: {0}'.format(sus_ev_1st))
        print('Remanining susceptible after 2nd outbreak: {0}'.format(sus_ev_2nd))
        print('Attacked in 1st outbreak: {0}'.format(pre_ev_1st))
        print('Attacked in 2nd outbreak: {0}'.format(pre_ev_2nd))
        print('Attacked in full epidemic: {0}'.format(pre_ev_fe))
        print('Deaths in 1st outbreak:{0}'.format(dea_ev_1st))
        print('Deaths in 2nd outbreak:{0}'.format(dea_ev_2nd))
        print('Deaths in full epidemic:{0}'.format(dea_ev_fe))
        print('Immunized in 1st outbreak: {0}'.format(imm_ev_1st))
        print('Immunized in 2nd outbreak: {0}'.format(imm_ev_2nd))
        print('Immunized in full epidemic: {0}'.format(imm_ev_fe))

        # Averted deaths
        diff_dea_1st = (dea_bl_1st - dea_ev_1st) * 1.0e+6
        diff_dea_1st = round(diff_dea_1st, 2)
        diff_dea_2nd = (dea_bl_2nd - dea_ev_2nd) * 1.0e+6
        diff_dea_2nd = round(diff_dea_2nd, 2)
        diff_dea_fe = (dea_bl_fe - dea_ev_fe) * 1.0e+6
        diff_dea_fe = round(diff_dea_fe, 2)

        # Diff. prevalence
        diff_pre_1st = (pre_bl_1st - pre_ev_1st) * 1.0e+6
        diff_pre_1st = round(diff_pre_1st, 2)
        diff_pre_2nd = (pre_bl_2nd - pre_ev_2nd) * 1.0e+6
        diff_pre_2nd = round(diff_pre_2nd, 2)
        diff_pre_fe = (pre_bl_fe - pre_ev_fe) * 1.0e+6
        diff_pre_fe = round(diff_pre_fe, 2)

        # Diff. immunity
        diff_imm_1st = (imm_bl_1st - imm_ev_1st) * 1.0e+6
        diff_imm_1st = round(diff_imm_1st, 2)
        diff_imm_2nd = (imm_bl_2nd - imm_ev_2nd) * 1.0e+6
        diff_imm_2nd = round(diff_imm_2nd, 2)
        diff_imm_fe = (imm_bl_fe - imm_ev_fe) * 1.0e+6
        diff_imm_fe = round(diff_imm_fe, 2)

        print('Models comparison')
        print('Attacked diff in 1st outbreak: {0}'.format(diff_pre_1st))
        print('Attacked diff in 2nd outbreak: {0}'.format(diff_pre_2nd))
        print('Attacked diff in full epidemic: {0}'.format(diff_pre_fe))
        print('Death diff in 1st outbreak: {0}'.format(diff_dea_1st))
        print('Death diff in 2nd outbreak: {0}'.format(diff_dea_2nd))
        print('Death diff in full epidemic: {0}'.format(diff_dea_fe))
        print('Immunized diff in 1st outbreak: {0}'.format(diff_imm_1st))
        print('Immunized diff in 2nd outbreak: {0}'.format(diff_imm_2nd))
        print('Immunized diff in full epidemic: {0}'.format(diff_imm_fe))

        # Plots
        #full_path = path + '/epivac_paper/results/comps'
        #total_results = res_bl, res_ev
        #pt.plot_vaccination_comparison(total_results, full_path)

        # Append to row list
        dealist.append([state, diff_dea_1st, diff_dea_2nd, diff_dea_fe])
        prelist.append([state, diff_pre_1st, diff_pre_2nd, diff_pre_fe])
        immlist.append([state, diff_imm_1st, diff_imm_2nd, diff_imm_fe])
        suslist.append([state, sus_bl_1st, sus_bl_2nd, sus_ev_1st, sus_ev_2nd])
        corlist.append([nva_bl_1st, diff_dea_1st, diff_dea_2nd])

    # Correlation
    dim = len(corlist)
    x = np.zeros(dim)
    y = np.zeros(dim)
    y2 = np.zeros(dim)
    
    for triple, i in zip(corlist, range(dim)):
        x[i] = triple[0]
        y[i] = triple[1]
        y2[i] = triple[2]

    x = x.reshape((-1, 1))
    model = LinearRegression().fit(x, y)
    model2 = LinearRegression().fit(x, y2)
    r_sq = model.score(x, y)
    r_sq2 = model2.score(x, y2)
    intercept = model.intercept_
    coef = model.coef_
    print(intercept)
    print(coef)
    print('R2 (1st) = {0}'.format(r_sq))
    print('R2 (2nd) = {0}'.format(r_sq2))

    # Generate death table
    with open(path + '/epivac_paper/results/table_deaths.csv', "w") as f1:
        wr = csv.writer(f1)
        wr.writerows(dealist)

    # Generate prevalence table
    with open(path + '/epivac_paper/results/table_prevalence.csv', "w") as f2:
        wr = csv.writer(f2)
        wr.writerows(prelist)

    # Generate immunized table
    with open(path + '/epivac_paper/results/table_immunized.csv', "w") as f3:
        wr = csv.writer(f3)
        wr.writerows(immlist)

    # Generate immunized table
    with open(path + '/epivac_paper/results/table_susceptible.csv', "w") as f4:
        wr = csv.writer(f4)
        wr.writerows(suslist)


def produce_figure_6():
    """ Produce figure 6.

    Figure 6 contains a scatter plot between attack rates during the second
    outbreak and the fraction of non-vaccinated individuals when every
    state has the same fraction of non-vaccinated individuals with respect
    to adult population. The median age of the state is highlighted.

    """
    
    # Prepare dictionary for state results
    global_results = {}

    # Loop over states
    for state in state_list:
    
        print('{0}'.format(state))

        # Obtain data
        update= True
        zeroprev = False
        counterfact = False
        mpars = feed_the_model(path, country, state, update, zeroprev, 
                               counterfact)
        
        # Vaccination campaign
        mpars['vac_id'] = 'pho'
        
        # Invoke Runge-Kutta
        mod_id = 'age_sir_vac'
        results = call_age_sir_vac(mpars)
    
        # Perform measurements
        results_1w = mo.measure_first_outbreak(results, mod_id)
        results_2w = mo.measure_second_outbreak(results, mod_id)
        results_full = mo.measure_full_epidemic(results, mod_id)

        # Store into global dictionary
        global_results[state] = {'1st': results_1w, 
                                 '2nd': results_2w,
                                 'full': results_full}

    # Plotting section
    full_path = path + '/epivac_paper/results'
    pt.plot_figure_6(global_results, full_path)


def produce_figure_7():
    """Produce figure 7. 
    
    A comparison of epidemic trajectories for two states in the age-SIR model
    with baseline vaccination and extra vaccination.

    """
    
    # Output dictionaries for plotting
    results_vac = {}
    results_exv = {}

    # General parameters
    state = 'Massachusetts'
    update = True # If True: 2019 census updated data
    zeroprev = False # If True: no initial seroprevalence at all
    counterfact = False
    mod_id = 'age_sir_vac' # Age-structured SIR with vaccination

    # Obtain data for state 1 in model with baseline vaccination
    mpars1 = feed_the_model(path, country, state, update, zeroprev, 
                            counterfact)

    # Vaccination campaign & 2nd outbreak
    mpars1['vac_id'] = 'pho'
    mpars1['extra_vac'] = 0.0
    #immunized_a = np.zeros(85)
    #immunized_a[18:] = 0.75 * mpars1['pop_age'][18:] 
    #immunized_a[0:18] = 0.3424012 * mpars1['pop_age'][0:18]
    #immunized_a = 0.4 * mpars1['pop_age']
    #mpars1['immunized'] = immunized_a # ADDED TO EXPLORE

    # Invoke Runge-Kutta & store results in model with baseline vaccination
    results_v1 = call_age_sir_vac(mpars1)
    results_vac[state] = results_v1

    # Compute & print some results
    res_fo = mo.measure_first_outbreak(results_v1, mod_id)
    res_so = mo.measure_second_outbreak(results_v1, mod_id)
    #res_fe = mo.measure_full_epidemic(results_v1, mod_id)

    # Extract results
    P_bl_1 = res_fo['pre_ss_reset'] * 1.0e+6
    D_bl_1 = res_fo['dea_reset'] * 1.0e+6
    P_bl_2 = res_so['pre_ss'] * 1.0e+6
    D_bl_2 = res_so['dea'] * 1.0e+6

    # Obtain data for state 2 in model with vaccination
    mpars2 = feed_the_model(path, country, state, update, zeroprev, 
                            counterfact)

    # Vaccination campaign & 2nd outbreak
    mpars2['vac_id'] = 'pho'
    mpars2['extra_vac'] = 0.01

    #immunized_a = np.zeros(85)
    #immunized_a[18:] = 0.42 * mpars2['pop_age'][18:]
    #immunized_a = 0.67 * mpars2['pop_age']
    #mpars2['immunized'] = immunized_a # ADDED TO EXPLORE

    # Invoke Runge-Kutta & store results in model with extra vaccination
    results_v2 = call_age_sir_vac(mpars2)
    results_exv[state] = results_v2
    
    # Compute & print some results
    res_fo = mo.measure_first_outbreak(results_v2, mod_id)
    res_so = mo.measure_second_outbreak(results_v2, mod_id)

    # Extract results
    P_ev_1 = res_fo['pre_ss_reset'] * 1.0e+6
    D_ev_1 = res_fo['dea_reset'] * 1.0e+6
    P_ev_2 = res_so['pre_ss'] * 1.0e+6
    D_ev_2 = res_so['dea'] * 1.0e+6
    
    # Model differences
    D_diff_1 = D_bl_1 - D_ev_1
    print(D_diff_1)
    P_diff_1 = P_bl_1 - P_ev_1
    print(P_diff_1)
    D_diff_2 = D_bl_2 - D_ev_2
    print(D_diff_2)
    P_diff_2 = P_bl_2 - P_ev_2
    print(P_diff_2)

    # Plot dynamics
    full_path = path + '/epivac_paper/results'
    total_results = results_vac, results_exv
    pt.plot_figure_7(total_results, full_path)


def main():
    """Launch the experiments that produce the figures in the draft.
    
    Figures 1, 2 and 3 are relatively fast to obtain. The rest may take a long
    while.

    """

    # Produce!
    #produce_phases()
    #produce_figure_1()
    #produce_figure_2()
    #produce_figure_3()
    produce_figures_4_and_5()
    #produce_tables()
    #produce_figure_6()

    print('Yarimashita!')


if __name__ == '__main__':

    main()