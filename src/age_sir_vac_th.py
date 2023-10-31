#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 17:50:19 2021

@author: alfonso

This file contains a data-driven age-structured epidemic SIR dynamics with
vaccination. In this version vaccination proceeds in a threshold-based
fashion.

The disease dynamics is based on Mistry et al. (2021):
    https://doi.org/10.1038/s41467-020-20544-y
An age-based susceptibility factor is added to 'control' infections within 
underage people.

Data sources can be checked in the data_utils.py file.

Model is integrated through 4-order Runge-Kutta numerical method and, if
conditions hold, at the end of every integration step a vaccination campaign
takes place where susceptible and recovered individuals are vaccinated 
proportionally and following real data from vaccination surveys
Lazer et al. (2021):
    Lazer, D., Ognyanova, K., Baum, M., Druckman, J., Green, J., Gitomer, 
    A., ... & Uslu, A. (2021). 
    The COVID States Project# 43: COVID-19 vaccine rates and attitudes 
    among Americans.

Dynamics proceeds in two phases:
    - Phase I: Restrictions measures are in place keeping a relatively low R0
    (under the typical/natural COVID-19 R0) but above the epidemic threshold.
    During this phase the threshold-based vaccination campaign takes place.
    - Phase II: Once the first outbreak is dies out, restrictions are lifted & 
    the system is minimally re-seeded, clearing the path to a second outbreak
    (depending on vaccination levels and state's population structure).

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
    
    chi_a = np.zeros(A, dtype=float)
    chi_a[0:20] = chi_ua # Maybe some ref. should be given to justify this value
    chi_a[20:] = 1.0
    
    return chi_a


def get_initial_vaccinated_by_age(vaccination_data):
    '''Extract people in the 'already vaccinated' category. '''
    
    return vaccination_data['vacpop'][0][:]


def get_initial_immunized_by_age(immunized_data, pop_a):
    
    return immunized_data * pop_a


def add_extra_vaccinated_by_age(vac_pop, pop_a, extra_vac=0.0):

    N_never_a = vac_pop[4][:]
    N_adult = np.sum(pop_a[18:])
    fraction = extra_vac / (np.sum(N_never_a) / N_adult)
    
    if fraction > 1.0:
        fraction = 1.0
    
    extravac_a = fraction * N_never_a
    
    return extravac_a


def get_not_vaccinated(vaccination_data):
    '''Extract all the individuals, by age class, that decline to uptake the
    COVID-19 vaccine.'''

    return vaccination_data['vacpop'][-1][:]


def set_initialization(pop_a, seeds_a, immunized_a, vaccinated_a):
    """Initialize the system for an age-structured SIR model with
    vaccination.
    
        Parameters
    ----------
    pop_a : np.array
        individuals in every age-class a
    seeds_a : np.array
        index cases in every age-class a
    immunized_a : np.array
        initially naturally immunized in every age-class a
    vaccinated_a : np.array
        initially vaccinated in every age-class a
    extra_vac : float
        extra fraction of vaccinated individuals (homogeneously shared in
        adult population)

    Returns
    -------
    pop_as : np.array
        number of individuals in every age-class a and every health-status s at
        t = 0

        pop[:,0]: susceptible individuals by age
        pop[:,1]: infected individuals by age
        pop[:,2]: recovered by age
        pop[:,3]: vaccinated by age
        pop[:,4]: naturally immunized by age (copy to not be subtracted when
        vaccinating)

    """

    pop = np.empty((len(pop_a), 5), dtype=float)
    
    # Compute extra vaccination
    extravac_a = np.zeros(len(pop_a), dtype=float)
    #extravac_a[18:] = extra_vac * pop_a[18:]

    # Proportional shares of vaccinated between susceptibles & naturally imm.
    susceptible_a = pop_a - seeds_a - immunized_a
    sus_vacc_a = susceptible_a * vaccinated_a / (immunized_a + susceptible_a)
    imm_vacc_a = immunized_a * vaccinated_a / (immunized_a + susceptible_a)

    pop[:,0] = susceptible_a - sus_vacc_a
    
    # Correct potentially negative population
    if np.any(pop[:,0] - extravac_a < 0) == True:
        negative_index = np.where(pop[:,0] - extravac_a < 0)[0]
        extravac_a[negative_index[0:len(negative_index)+1]] = \
        pop[negative_index[0:len(negative_index)+1], 0]

    pop[:,0] -= extravac_a

    pop[:,1] = seeds_a
    
    pop[:,2] = immunized_a - imm_vacc_a
    pop[:,3] = vaccinated_a + extravac_a
    pop[:,4] = immunized_a

    if np.any(pop[:,0]<0) == True:
        print('NEGATIVE SUSCEPTIBLE')
    if np.any(pop[:,2]<0) == True:
        print('NEGATIVE NAT. IMMUNIZED')

    return pop


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
    
    lambda_ = np.max(np.linalg.eigvals(chi_a * contact)) # original computation
    mpars['lambda'] = lambda_
    beta = (R0 * gamma) / lambda_ 

    return beta
    

def age_structured_sir_dynamics(pop_as, mpars):
    """Age-structured SIR model dynamical equations. A mass-action
    (frequency dependent) force of infection and age-dependent susceptibility
    factor is used.
    
    For a reference of a model similar to this one see Mistry (2021). For more 
    details on the nature of the mass-action term see Keeling & Rohani (2011).
    
    Additionally a susceptibility factor is introduced
    
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
        
        eq_array[:,0]: change in susceptibles by age-class a
        eq_array[:,1]: change in infecteds by age-class a
        eq_array[:,2]: change in recovereds by age-class a
        eq_array[:,3]: change in recovereds by age-class a (added because
        population from slot 2 will be substracted when vaccinating)        

    """

    # Unpack parameters
    beta = mpars['beta']
    gamma = mpars['gamma']
    contact = mpars['contact']
    pop_a = mpars['pop_age']
    chi_a = mpars['chi_a']

    N = np.sum(pop_a)
    A = len(pop_a)
    
    # Compute force of infection per age-class a
    foi_a = beta * N * (contact.dot(pop_as[:,1] / pop_a)) * chi_a

    # Compute r.h.s. for every S-I-R health statuses and A age-classes
    eq_array = np.zeros((A, 5), dtype=float)
    eq_array[:,0] = - foi_a * pop_as[:,0]
    eq_array[:,1] = foi_a * pop_as[:,0] - gamma * pop_as[:,1]
    eq_array[:,2] = gamma * pop_as[:,1]
    eq_array[:,4] = eq_array[:,2].copy()

    return eq_array


def threshold_vaccination_campaign(pop_as, h, mpars, vpop):
    """Vaccinate a certain fraction of individuals per age-class a by a
    threshold-based procedure. Thresholds are somewhat arbitrary based on the
    categories in the survey of Lazer (2021).
    
    - Phase 1: Vaccinate fraction (N_a*f_1,a)/t_vac per day until 0.2
    - Phase 2: Vaccinate fraction N_a*(f_1,a+f_2,a)/t_vac,a per day until 0.5
    - Phase 3: Vaccinate fraction N_a*(f_1,a+f_2,a+f_3,a)/t_vac per day until
    finishing

    Parameters
    ----------
    pop_as : np.array
        population in age-class a and health-status s
    pop_vac : np.array
        vaccinated population in age-class a
    mpars : dict
        includes population by age, fraction of people to be vaccinated, 
        duration of vaccination campaign
        
    Returns
    -------
    pop: np.array
        population in age-class a and health-status s

    """

    # Unpack parameters
    pop_a = mpars['pop_age'] / mpars['N']
    
    # Total susceptible & removed per age-class pool
    vac_pool_a = pop_as[:,0] + pop_as[:,2]

    # Threshold procedure
    if (np.sum(pop_as[:,3]) < 0.2 * np.sum(pop_a[18:])):

        vacc_p1 = mpars['vacc_t1']

        vacc = vacc_p1 * h
        
        sus_vac_a = vacc * (pop_as[:,0] / vac_pool_a)
        rem_vac_a = vacc * (pop_as[:,2] / vac_pool_a)

    elif (np.sum(pop_as[:,3]) < 0.5 * np.sum(pop_a[18:])):

        vacc_p1 = mpars['vacc_t1']
        vacc_p2 = mpars['vacc_t2']

        vacc = (vacc_p1 + vacc_p2) * h
        
        sus_vac_a = vacc * (pop_as[:,0] / vac_pool_a)
        rem_vac_a = vacc * (pop_as[:,2] / vac_pool_a)

    elif (np.sum(pop_as[:,3]) >= 0.5 * np.sum(pop_a[18:])):

        vacc_p1 = mpars['vacc_t1']
        vacc_p2 = mpars['vacc_t2']
        vacc_p3 = mpars['vacc_t3']

        vacc = (vacc_p1 + vacc_p2 + vacc_p3) * h
        
        sus_vac_a = vacc * (pop_as[:,0] / vac_pool_a)
        rem_vac_a = vacc * (pop_as[:,2] / vac_pool_a)
        
    # Correct potential negative population for susceptibles
    if np.any(pop_as[:,0] - sus_vac_a < 0) == True:
        negative_index = np.where(pop_as[:,0] < sus_vac_a)[0]
        sus_vac_a[negative_index[0:len(negative_index)+1]] = \
        pop_as[negative_index[0:len(negative_index)+1], 0]
        
    # Correct potential negative population for removed
    if np.any(pop_as[:,2] - rem_vac_a < 0) == True:
        negative_index = np.where(pop_as[:,2] < rem_vac_a)[0]
        rem_vac_a[negative_index[0:len(negative_index)+1]] = \
        pop_as[negative_index[0:len(negative_index)+1], 2]
    
    # Update populations
    pop_as[:,0] -= sus_vac_a
    pop_as[:,2] -= rem_vac_a
    pop_as[:,3] += (sus_vac_a + rem_vac_a) #vaccinated_a

    return pop_as


def correct_populations(pop_as, A):
    
    pop_as[:,2] += pop_as[:,1]
    pop_as[:,4] += pop_as[:,1]
    pop_as[:,1] = np.zeros(A, dtype=float)
    
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
    
    new_seeds_a = np.zeros(A, dtype=float)
    new_seeds_a[18:60] = 1.0e-03 * pop_as[18:60].T[0]

    pop_as[:,0] -= new_seeds_a
    pop_as[:,1] += new_seeds_a

    return pop_as    


def model_integration(pop_as0, t_max, h, mpars):

    # Prepare final results dictionary
    output = {'1st': {}, '2nd': {}, 'full': {}}
    
    # Prepare some initializations
    i_max = int(t_max/h)

    first_wave = True
    second_wave = mpars['second_wave']
    t_2w = 700 # Arbitrary time (days) to start a 2nd wave/outbreak
    infected_threshold = 1.0 / (1000.0 * mpars['N'])

    if second_wave == False:
        t_max = t_2w
        i_max = int(t_max/h)
    
    dyn_eqs = age_structured_sir_dynamics # Type of dynamical equations
    
    A = np.shape(pop_as0)[0]
    H = np.shape(pop_as0)[1]
    pop = np.empty((i_max, A, H), dtype=float) # t-series for each age-class & health-status
    pop[0,:] = pop_as0.copy()
    
    # Set vaccination campaign
    pop_a = mpars['pop_age'] / mpars['N']
    if 'camp' in mpars:
        vaccination_campaign = mpars['camp']
        vac = True
    else:
        vac = False
        
    extra_vac = mpars['extra_vac']
    extravac_a = add_extra_vaccinated_by_age(mpars['vaccination']['vacpop'],
                                             pop_a, extra_vac)

    t_start = mpars['t_start']
    t_vac = mpars['t_vac']
    i_start = int(t_start / h)
    i_vac = int((t_vac + t_start) / h)
    
    # Download populations in vaccination categories
    vpop = np.zeros((5, 85))
    vpop[0,:] = mpars['vaccination']['vacpop'][0,:] / mpars['N']
    vpop[1,:] = mpars['vaccination']['vacpop'][1,:] / mpars['N'] + extravac_a
    vpop[2,:] = mpars['vaccination']['vacpop'][2,:] / mpars['N']
    vpop[3,:] = mpars['vaccination']['vacpop'][3,:] / mpars['N']
    vpop[4,:] = mpars['vaccination']['vacpop'][4,:] / mpars['N'] - extravac_a
    
    # Set vaccination flow for individuals in 'soon' category
    vacc_p1_a = vpop[1,:] / t_vac
    mpars['vacc_t1'] = vacc_p1_a
    
    # Set starting time for phase II
    t_vac12 = (0.2 * np.sum(pop_a[18:]) - np.sum(vpop[0,:])) / np.sum(vacc_p1_a)
    if t_vac12 < 0:
        t_vac12 = 0
        
    # Set vaccination flow for individuals in 'some' category
    t_vac2 = t_vac - t_vac12
    vacc_p2_a = vpop[2,:] / t_vac2
    mpars['vacc_t2'] = vacc_p2_a
    
    # Set starting time for phase III
    t_vac23 = (0.5 * np.sum(pop_a[18:]) - np.sum(vpop[0,:])) / np.sum(vacc_p2_a)
    if t_vac23 < 0:
        t_vac23 = 0

    # Set vaccination flow for individuals in 'almost' category
    t_vac3 = t_vac - t_vac23
    vacc_p3_a = vpop[3,:] / t_vac3
    mpars['vacc_t3'] = vacc_p3_a

    # Loop over integration time (index)
    t = 0
    t_array = np.linspace(0, t_max, i_max)
    
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
        
        # Vaccination campaign during the 1st outbreak
        if ((i >= i_start) and (i <= (i_vac + i_start)) and vac == True):
            pop[i+1] = vaccination_campaign(pop[i+1], h, mpars, vpop)

        # Collect 1st outbreak results and...
        current_infected = np.sum(pop[i+1].T[1])
        if (current_infected < infected_threshold) and first_wave == True: 

            # Collect some data at the end of the 1st outbreak
            output['1st']['pop'] = pop[0:i+1]
            i_end = i
            output['1st']['i_end'] = i_end
            output['1st']['t_end'] = t
            output['1st']['time'] = t_array[0:i_end+1]

            first_wave = False # do not enter here again

            # CONTROL PRINT
            print(' \n End of the first outbreak')
            i1_tot = np.sum(pop[i+1][:].T[1])
            print('Infected density: {0}'.format(i1_tot))
            v1_tot = np.sum(pop[i+1][:].T[3])
            print('Vaccinated density: {0}'.format(v1_tot))
            r1_tot = np.sum(pop[i+1][:].T[2])
            print('Removed (not vac.) density: {0}'.format(r1_tot))
            s1_tot = np.sum(pop[i+1][:].T[0])
            print('Susceptible density: {0}'.format(s1_tot))

            # ... Reseed & prepare conditions for a 2nd outbreak
            if second_wave == True:

                print('2nd outbreak starts at t={0}'.format(t))

                # Compute new beta under new R0
                mpars['beta'] = extract_beta_from_R0(mpars, 2)

                # Smash residual infecteds
                correct_populations(pop[i+1], A)

                # Reseed
                reseeded_pop = reseed_epidemic(pop[i+1], A)
                pop[i+1] = reseeded_pop

                # CONTROL PRINT
                print(' \n After re-seeding')
                i1_tot = np.sum(pop[i+1][:].T[1])
                print('Infected density: {0}'.format(i1_tot))
                v1_tot = np.sum(pop[i+1][:].T[3])
                print('Vaccinated density: {0}'.format(v1_tot))
                r1_tot = np.sum(pop[i+1][:].T[2])
                print('Removed (not vac.) density: {0}'.format(r1_tot))
                s1_tot = np.sum(pop[i+1][:].T[0])
                print('Susceptible density: {0}'.format(s1_tot))
                s1_underage = np.sum(pop[i+1][0:18].T[0]) / s1_tot
                print('Underage susceptible: {0}'.format(s1_underage))
                s1_young = np.sum(pop[i][18:45].T[0]) / s1_tot
                print('Young susceptible: {0}'.format(s1_young))
                s1_mature = np.sum(pop[i+1][45:65].T[0]) / s1_tot
                print('Medium-Mature susceptible: {0}'.format(s1_mature))
                s1_elder = np.sum(pop[i+1][65:].T[0]) / s1_tot
                print('Elder susceptible: {0}'.format(s1_elder))

        t += h

    print('\n End of the spreading')
    
    # CONTROL PRINT
    i2_tot = np.sum(pop[i+1][:].T[1])
    print('Infected density: {0}'.format(i2_tot))
    v2_tot = np.sum(pop[i+1][:].T[3]) - v1_tot
    print('Vaccinated density: {0}'.format(v2_tot))
    r2_tot = np.sum(pop[i+1][:].T[2]) - r1_tot
    print('Removed (not vac.) density: {0}'.format(r2_tot))
    s2_tot = np.sum(pop[i+1][:].T[0]) - s1_tot
    print('Susceptible density: {0}'.format(s2_tot))
    #s_underage = np.sum(pop[i+1][0:18].T[0]) / s2_tot
    #print('Underage susceptible: {0}'.format(s_underage))
    #s_young = np.sum(pop[i][18:45].T[0]) / s2_tot
    #print('Young susceptible: {0}'.format(s_young))
    #s_mature = np.sum(pop[i+1][45:65].T[0]) / s2_tot
    #print('Medium-Mature susceptible: {0}'.format(s_mature))
    #s_elder = np.sum(pop[i+1][65:].T[0]) / s2_tot
    #print('Elder susceptible: {0}'.format(s_elder))
    
    # Analytical solution
    sus0_a = pop[i_end][:].T[0]
    sus_ss_a = pop[-1][:].T[0]
    rem0_a = pop[i_end][:].T[2] + pop[i_end][:].T[3]
    contact = mpars['contact']
    chi_a = mpars['chi_a']
    pop_a = mpars['pop_age'] 
    beta = mpars['beta']
    gamma = mpars['gamma']
    N = np.sum(pop_a)
    cum_inf_a = (pop_a / N) - rem0_a - sus_ss_a
    arg = np.zeros(len(pop_a), dtype=float)
    arg = -beta * N * (contact.dot(cum_inf_a / pop_a)) * chi_a / gamma
    rhs = np.exp(arg)
    lhs = (sus_ss_a / sus0_a)
    print('Difference: {0}'.format(lhs - rhs))

    # Collect some data from the 2nd outbreak
    if second_wave == True:

        output['2nd']['pop'] = pop[i_end+1:]
        output['2nd']['i_end'] = i+1
        output['2nd']['t_end'] = t
        output['2nd']['time'] = t_array[i_end+1:] 

    # Collect full epidemic data
    output['full']['time'] = t_array
    output['full']['pop'] = pop

    return output

    return output


def call_age_sir_vac_thr(mpars):
    """Call the age-structured SIR dynamics with threshold-based vaccination.
    
    Model is integrated through 4-order Runge-Kutta numerical method.
    More details of the model in Mistry et al. (2021):
        https://doi.org/10.1038/s41467-020-20544-y
    An age-based susceptibility factor is added to 'control' infections
    within underage people.
    
    Vaccination proceeds in a 'manual' way, not as part of the dynamical
    equations. While the vaccination campaign is open, at the end of every
    integration step susceptible and recovered individuals are proportionally
    vaccinated, following a threshold-based scheme with real data from surveys 
    for each state and age class.

    Parameters
    ----------
    mpars : dict
        contains model data and parameters
    
    Returns
    -------
    output : dict
        contains model results, population time series & relevant parameters
    
    """
    
    print('Spreading for {0}'.format(mpars['state']))

    # Output dictionary for further analysis & plotting
    output = {}

    # Normalize populations
    N = np.sum(mpars['pop_age'])
    mpars['N'] = N
    pop_a = mpars['pop_age'] / N
    seeds_a = mpars['seeds'] / N
    immunized_a = mpars['immunized'] / N

    # Introduce some age-susceptibility to contagion
    mpars['chi_a'] = set_age_susceptibility_factor(len(pop_a))

    # SET VACCINATION CAMPAIGN
    if 'vac_id' in mpars.keys():

        # Extra-vaccination efforts
        if 'extra_vac' not in mpars:
            mpars['extra_vac'] = 0.0

        # Initial condition for vaccinated
        vaccinated_a = get_initial_vaccinated_by_age(mpars['vaccination']) / N

        # Type of vaccination campaign
        vaccination_campaign = {'thr': threshold_vaccination_campaign}
        vac_id = mpars['vac_id']
        mpars['camp'] = vaccination_campaign[vac_id]

    else:

        vaccinated_a = np.zeros(len(pop_a), dtype=float) # No initial vaccinated at all

    # Runge-Kutta algorithm parameters
    h = 1.0 / 24.0 # integration step (hours)
    t_max = 1500 # days
    
    # Transmission rate
    mpars['beta'] = extract_beta_from_R0(mpars)

    # INITIAL CONDITIONS
    pop_0 = set_initialization(pop_a, seeds_a, immunized_a, vaccinated_a)

    # INVOKE RUNGE-KUTTA
    output['age_sir_vac_thr'] = model_integration(pop_0, t_max, h, mpars)

    # Add some relevant data to results dictionary for further manipulations
    if 'extra_vac' in mpars.keys():
        output['extra_vac'] = mpars['extra_vac']
    output['country'] = mpars['country']
    output['state'] = mpars['state']
    output['pop_age'] = pop_a
    output['tot_pop'] = np.sum(pop_a)
    output['deaths'] = mpars['deaths']
    output['lambda'] = mpars['lambda']

    return output


def main():
    '''Launch the model for a particular state and epidemic conditions.
    Perform some calculations and make some plots.'''

    # SET TERRITORY & BASIC SETTINGS
    country = 'United_States'
    state = 'Oklahoma'
    update = True # if True: 2019 CENSUS DATA
    zeroprev = False # if True: NO SERO-PREVALENCE AT ALL

    # EXTRACT INPUT DATA
    mpars = feed_the_model(path, country, state, update, zeroprev)

    # SET EPIDEMIC PARAMETERS
    mpars['R0'] = 1.5
    mpars['R0_P2'] = 3.0
    mpars['chi_ua'] = 0.56

    # SET VACCINATION PARAMETERS
    mpars['vac_id'] = 'thr'
    mpars['extra_vac'] = 0.0

    #immunized_a = np.zeros(85)
    #immunized_a[18:] = 0.75 * mpars['pop_age'][18:] 
    #immunized_a[0:18] = 0.33 * mpars['pop_age'][0:18]
    #immunized_a = 0.67 * mpars['pop_age']
    #mpars['immunized'] = immunized_a # ADDED TO EXPLORE

    # CALL THE MODEL
    results = call_age_sir_vac_thr(mpars)

    # OBTAIN OBSERVABLES
    mod_id = 'age_sir_vac_thr'
    fo_dict = mo.measure_first_outbreak(results, mod_id)
    so_dict = mo.measure_second_outbreak(results, mod_id)
    fe_dict = mo.measure_full_epidemic(results, mod_id)

    # PLOT RESULTS
    full_path = path + '/epivac_paper/results'
    pt.plot_dynamics(results, mod_id, full_path)


if __name__ == '__main__':
    
    main()
