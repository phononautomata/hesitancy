#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 17:50:19 2021

@author: alfonso

This file contains a data-driven age-structured epidemic SIR dynamics.
The model is based on Mistry et al. (2021):
    https://doi.org/10.1038/s41467-020-20544-y
An age-based susceptibility factor is added to 'control' infections within 
underage people.

Data sources can be checked in the data_utils.py file. In this model there is
no vaccination campaign and thus no data from vaccination.

Model is integrated through 4-order Runge-Kutta numerical method.

Dynamics proceeds in two phases:
    - Phase I: Restrictions measures are in place keeping a relatively low R0
    (under the typical/natural COVID-19 R0) but above the epidemic threshold.
    - Phase II: Once the first outbreak is dies out, restrictions are lifted & 
    the system is minimally re-seeded, clearing the path to a second outbreak.

"""

import numpy as np

from data_utils import feed_the_model
import measure_outbreaks as mo
import plots as pt

path = '/Users/ademiguel/Workshop'


def set_age_susceptibility_factor(A, chi_ua=0.56):
    """Set an age-based infection susceptibility factor to avoid over-contagion 
    among underage.
    
    Parameters
    ----------
    A : int
        number of age-classes
    chi_ua : float (optional)
        susceptibility factor for underage
    
    Returns
    -------
    chi_a : np.array
        age-based infection susceptibility factor
    """
    
    chi_a = np.zeros(A)
    chi_a[0:20] = chi_ua # Maybe some ref. should be given to justify this value
    chi_a[20:] = 1.0
    
    return chi_a


def set_initialization(pop_a, seeds_a, immunized_a):
    """Initialize the system for an age-structured model.
    
    Parameters
    ----------
    pop_a : np.array
        individuals in every age-class a
    seeds_a : np.array
        index cases in every age-class a
    immunized_a : np.array
        initially naturally immunized in every age-class a

    Returns
    -------
    pop_as : np.array
        number of individuals in every age-class a and every health-status s at
        t = 0
        pop_as[:,0]: susceptible individuals by age
        pop_as[:,1]: infected individuals by age
        pop_as[:,2]: naturally immunized by age

    """

    pop_as = np.empty((len(pop_a), 3), dtype=float)

    pop_as[:,0] = pop_a - seeds_a - immunized_a
    pop_as[:,1] = seeds_a
    pop_as[:,2] = immunized_a

    return pop_as


def extract_beta_from_R0(mpars, phase=None):
    """Extract beta, transmission rate, from the basic reproductive number R0.
    
    The computation is done by implying the largest eigenvalue of the contact
    matrix. BEWARE: the age susceptibility factor must be included too.
    
    See Diekmann (1990) for more: https://doi.org/10.1007/BF00178324
    
    Parameters
    ----------
    mpars : dict
        model parameters
    phase : int (optional)
        phase of the proccess in case there is a different R0 for each
    
    Returns
    -------
    beta : float
        transmission rate

    """

    gamma = mpars['gamma']
    contact = mpars['contact']
    chi_a = mpars['chi_a']

    if phase == 2:
        R0 = mpars['R0_P2'] # R0 when entering the second outbreak
    else:
        R0 = mpars['R0']

    lambda_ = np.max(np.linalg.eigvals(chi_a * contact))
    beta = (R0 * gamma) / lambda_ 

    return beta


def age_structured_sir_dynamics(pop_as, mpars):
    """Age-structured SIR model dynamical equations. 
    
    A mass-action (frequency dependent) force of infection and age-dependent 
    susceptibility factor is used.
    
    For a reference of a model similar to this one see Mistry (2021). For more 
    details on the nature of the mass-action term see Keeling & Rohani (2011).

    Parameters
    ----------
    pop_as : np.array
        population in age-class a and health-status s
    mpars : dict
        model parameters: beta, gamma, contact matrix, population by age-class

    Returns
    -------
    eq_array : np.array
        r.h.s. of the dynamical equations of the model for all age-classes and
        health-statuses

    """

    beta = mpars['beta']
    gamma = mpars['gamma']
    contact = mpars['contact']
    pop_a = mpars['pop_age']
    chi_a = mpars['chi_a']

    N = np.sum(pop_a)
    A = len(pop_a)

    # Compute r.h.s. for every S-I-R health statuses and A age-classes
    eq_array = np.zeros((A, 3))
    eq_array[:,0] = -beta * N * (contact.dot(pop_as[:,1]/ pop_a)) * pop_as[:,0] \
                    * chi_a
    eq_array[:,1] = beta * N * (contact.dot(pop_as[:,1] / pop_a)) * pop_as[:,0] \
                    * chi_a - gamma * pop_as[:,1]
    eq_array[:,2] = gamma * pop_as[:,1]

    return eq_array


def correct_populations(pop_as, A):
    
    pop_as[:,2] += pop_as[:,1]
    pop_as[:,4] += pop_as[:,1]
    pop_as[:,1] = np.zeros(A)
    
    return pop_as


def reseed_epidemic(pop_as, A):
    """Introduce new index cases.
    
    New seeds suppose a certain fraction of the remanining adult susceptible
    population at the end of the first outbreak. This could be seen as somehow
    arbitrary to be honest, indeed.
    
    Parameters
    ----------
    pop_as : np.array
        individuals in every age-class a and health-status s
    A : int
        number of age-classes

    Returns
    -------
    pop_as : np.array
        individuals in every age-class a and health-status s

    """
    
    new_seeds_a = np.zeros(A)
    new_seeds_a[18:60] = 1.0e-03 * pop_as[18:60].T[0]

    pop_as[:,0] -= new_seeds_a
    pop_as[:,1] += new_seeds_a

    return pop_as   


def model_integration(pop_as0, t_max, h, mpars):
    """Integration of the age-structured SIR dynamical equations through
    the 4th order Runge-Kutta numerical method.
    
    This model includes the possibility of a second outbreak.
    
    Parameters
    ----------
    pop_as0 : np.array
        individuals in every age-class a
    t_max : float
        maximum integration time
    h : float
        integration step
    mpars : dict
        contains model parameters and other stuff
        
    Returns
    -------
    output : dict
        contains population time series for every age-class and health-status
        during the first outbreak, second outbreak, and full epidemic

    """

    # Prepare final results dictionary
    output = {'1st': {}, '2nd': {}, 'full': {}}

    # Prepare some initializations
    i_max = int(t_max/h)

    first_wave = True
    second_wave = mpars['second_wave']
    t_2w = 500 # Arbitrary time (days) to start a 2nd wave/outbreak
    infected_threshold = 1.0 / (1000.0 * mpars['N'])

    if second_wave == False:
        t_max = t_2w
        i_max = int(t_max/h)

    dyn_eqs = age_structured_sir_dynamics

    A = np.shape(pop_as0)[0]
    H = np.shape(pop_as0)[1]
    pop = np.empty((i_max, A, H)) # t-series for each age-class & health-status
    pop[0,:] = pop_as0.copy()

    # Loop over integration time (index)
    t = 0
    t_array = np.linspace(0, t_max, i_max)
    print('1st outbreak starts')
    
    for i in range(0, i_max-1):
        
        # Compute coefficients for all state variables
        k1 = h * dyn_eqs(pop[i], mpars)
        k2 = h * dyn_eqs(pop[i] + 0.5 * k1, mpars)
        k3 = h * dyn_eqs(pop[i] + 0.5 * k2, mpars)
        k4 = h * dyn_eqs(pop[i] + k3, mpars)
        
        # 4th order Runge-Kutta integration step
        change = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        # Compute function at next time step (integration)
        pop[i+1] = np.add(pop[i], change)

        # Collect 1st outbreak results and...
        current_infected = np.sum(pop[i+1].T[1])
        if (current_infected < infected_threshold) and first_wave == True:
            
            # Collect data at the end of the 1st outbreak
            output['1st']['pop'] = pop[0:i+1]
            i_end = i
            output['1st']['i_end'] = i_end
            output['1st']['t_end'] = t
            output['1st']['time'] = t_array[0:i_end+1]

            first_wave = False # do not enter here again
            
            # CONTROL PRINT
            s_tot = np.sum(pop[i+1][:].T[0])
            s_underage = np.sum(pop[i+1][0:18].T[0]) / s_tot
            print('Underage susceptible: {0}'.format(s_underage))
            s_young = np.sum(pop[i+1][18:45].T[0]) / s_tot
            print('Young susceptible: {0}'.format(s_young))
            s_mature = np.sum(pop[i+1][45:65].T[0]) / s_tot
            print('Medium-Mature susceptible: {0}'.format(s_mature))
            s_elder = np.sum(pop[i+1][65:].T[0]) / s_tot
            print('Elder susceptible: {0}'.format(s_elder))

            # ... Reseed & prepare conditions for a 2nd outbreak
            if second_wave == True:

                print('2nd outbreak starts at t={0}'.format(t))

                # Compute new beta under new R0
                mpars['beta'] = extract_beta_from_R0(mpars, 2)
                
                # Smash residual infecteds
                correct_populations(pop[i+1], A)
            
                # Reseed
                pop[i+1] = reseed_epidemic(pop[i+1], A)
                
                # CONTROL PRINT
                print('After reseeding')
                s_tot = np.sum(pop[i+1][:].T[0])
                s_underage = np.sum(pop[i+1][0:18].T[0]) / s_tot
                print('Underage susceptible: {0}'.format(s_underage))
                s_young = np.sum(pop[i+1][18:45].T[0]) / s_tot
                print('Young susceptible: {0}'.format(s_young))
                s_mature = np.sum(pop[i+1][45:65].T[0]) / s_tot
                print('Medium-Mature susceptible: {0}'.format(s_mature))
                s_elder = np.sum(pop[i+1][65:].T[0]) / s_tot
                print('Elder susceptible: {0}'.format(s_elder))

        t += h
        
    print('End of the spreading')

    # Collect some data from the 2nd outbreak if applies
    if second_wave == True:

        output['2nd']['pop'] = pop[i_end:]
        output['2nd']['i_end'] = i+1
        output['2nd']['t_end'] = t
        output['2nd']['time'] = t_array[i_end:]
    
    # Collect full epidemic data
    output['full']['time'] = t_array
    output['full']['pop'] = pop

    return output


def call_age_sir(mpars):
    """Call the age-structured SIR dynamics (no vaccination here).
    
    Model is integrated through 4-order Runge-Kutta numerical method.
    More details of the model in Mistry et al. (2021):
        https://doi.org/10.1038/s41467-020-20544-y
    An age-based susceptibility factor is added to 'control' infections
    within underage people.
    
    Parameters
    ----------
    mpars : dict
        contains model data and parameters
    
    Results
    -------
    results : dict
        contains model results, population time series & relevant parameters

    """
    
    print('Spreading for {0}'.format(mpars['state']))

    output = {}

    # Normalize populations
    N = np.sum(mpars['pop_age'])
    mpars['N'] = N
    pop_a = mpars['pop_age'] / N
    seeds_a = mpars['seeds'] / N
    immunized_a = mpars['immunized'] / N
    mpars['chi_a'] = set_age_susceptibility_factor(len(pop_a), mpars['chi_ua'])

    # Runge-Kutta algorithm parameters
    h = 1.0 / 24.0 # integration step (hours^{-1})
    t_max = 1000 # days

    # Transmission rate
    mpars['beta'] = extract_beta_from_R0(mpars)

    # Initial conditions
    pop_0 = set_initialization(pop_a, seeds_a, immunized_a)

    # Invoke Runge-Kutta (like if it were some kind of ancestral demon)
    output['age_sir'] = model_integration(pop_0, t_max, h, mpars)

    # Add some relevant data to results dictionary for further manipulations
    output['country'] = mpars['country']
    output['state'] = mpars['state']
    output['pop_age'] = pop_a
    output['tot_pop'] = np.sum(pop_a)
    output['deaths'] = mpars['deaths']

    return output


def main():
    '''Launch the model for a particular state and epidemic conditions.
    Perform some calculations and make some plots.'''

    # Territory
    country = 'United_States'
    state = 'Massachusetts'

    # EXTRACT INPUT DATA
    update = True
    zeroprev = True
    mpars = feed_the_model(path, country, state, update, zeroprev)

    # SET EPIDEMIC PARAMETERS
    mpars['R0'] = 1.5
    mpars['R0_P2'] = 3.0
    mpars['chi_ua'] = 0.56
    mpars['second_wave'] = True

    # CALL THE MODEL
    results = call_age_sir(mpars)

    # OBTAIN OBSERVABLES
    mod_id = 'age_sir'
    
    #fo_dict = mo.measure_first_outbreak(results, mod_id)
    #so_dict = mo.measure_second_outbreak(results, mod_id)
    #fe_dict = mo.measure_full_epidemic(results, mod_id)
    
    full_path = path + '/epivac_paper/results'
    pt.plot_dynamics(results, mod_id, full_path)


if __name__ == '__main__':
    
    main()