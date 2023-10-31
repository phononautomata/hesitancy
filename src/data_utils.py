#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 22:48:16 2021

@author: alfonso

This file contains some utility functions related mainly to the process of
downloading, importing data, store and transform it properly, and the like. 

"""

import requests
import urllib.request
import time
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd


def download_census_data(path):
    """Download US census data from 2019 for every state and single age groups.
    
    Age groups are given from age 0 to age 83, and then 84+.

    Parameters
    ----------
    path : str
        where the data is to be stored

    Returns
    -------
    nothing

    """
    
    path = '/Users/ademiguel/Workshop/'

    state_list = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 
              'Colorado', 'Connecticut', 'Delaware', 'District_of_Columbia', 
              'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana',
              'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
              'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 
              'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New_Hampshire',  
              'New_Jersey', 'New_Mexico', 'New_York', 'North_Carolina', 
              'North_Dakota', 'Ohio', 'Oklahoma', 'Oregon', 
              'Pennsylvania', 'Rhode_Island', 'South_Carolina', 'South_Dakota', 
              'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 
              'Washington', 'West_Virginia', 'Wisconsin', 'Wyoming']
    
    # URL to webscrape from
    url = 'https://www.census.gov/data/tables/time-series/demo/popest/2010s-state-detail.html'
    
    # Connect to the URL
    response = requests.get(url)
    # Parse HTML and save to BeautifulSoup object
    soup = BeautifulSoup(response.text, "html.parser")

    # Download full dataset
    to_be_downloaded = soup.findAll('a')[288:339]
    for one_a_tag, state in zip(to_be_downloaded, state_list):
        
        link = one_a_tag['href']
        download_url = 'https:' + link

        filename = '2019_United_States_subnational_' + state + '_age_distribution_85' + '.xlsx'
        stored_data = path + '/epivac_paper/data/' + filename

        urllib.request.urlretrieve(download_url, stored_data)

        time.sleep(1)


def import_age_distribution(path, country, state):
    """Import age distribution of a territory from a csv file and store it into 
    an array.
    
    Original data from https://github.com/mobs-lab/mixing-patterns

    Parameters
    ----------
    path : str
        path where the csv file lies
    country : str
        name of the country
    state : str
        name of the region/state

    Returns
    -------
    - : np.array
        population distribution by age class
    
    """

    lower_path = '/epivac_paper/data/'
    file_name = country + '_subnational_' + state + '_age_distribution_85'
    full_name = path + lower_path + file_name + '.csv'

    age_df = pd.read_csv(full_name, header=None)
    #print(np.sum(age_df.values.T[1]))
    
    pop_a = np.zeros(len(age_df.values.T[1]), dtype=float)
    pop_a = age_df.values.T[1]

    return pop_a
    

def import_updated_age_distribution(path, country, state):
    """Import updated age distribution of a territory from a xlsl file and
    store it into an array.
    
    Data from 2019 census includes a category of 84 y.o. and then 85+. Data
    from 2018 seems to include just a category 84+. To adapt it, the last
    two age classes of 2019 are merged into one of 84+.
    
    Parameters
    ----------
    path : str
        path where the csv file lies
    country : str
        name of the country
    state : str
        name of the region/state
        
    Returns
    -------
    - : np.array
        population distribution by age class

    """

    lower_path = '/epivac_paper/data/'
    file_name = '2019_' + country + '_subnational_' + state + '_age_distribution_85'
    full_name = path + lower_path + file_name + '.xlsx'

    full_age_df = pd.read_excel(full_name)

    age_df = full_age_df['Unnamed: 34'][5:90].values
    merge_older = full_age_df['Unnamed: 34'][89] + full_age_df['Unnamed: 34'][90]
    age_df[-1] = merge_older
    
    pop_a = np.zeros(len(age_df), dtype=float)
    for a in range(len(age_df)):
        pop_a[a] = age_df[a]

    return pop_a


def compute_median_age(pop_a):
    
    summa_pop = 0
    
    A = len(pop_a)
    
    N = np.sum(pop_a)
    
    for age in range(A):
        summa_pop += pop_a[age]
        
        if summa_pop >= N/2.0:
            break
    
    median = (age + (age+1)) / 2.0
    
    return median


def obtain_median_age(state):
    
    median_age_dict = {'Alaska': 35.0, 'Alabama': 39.4, 'Arkansas': 38.5, 
                       'Arizona': 38.2, 'California': 37.0, 'Colorado': 37.1,
                       'Connecticut': 41.1, 'District_of_Columbia': 34.2,
                       'Delaware': 41.1, 'Florida': 42.5, 'Georgia': 37.1, 
                       'Hawaii': 39.6, 'Iowa': 38.5, 'Idaho': 36.9,
                       'Illinois': 38.6, 'Indiana': 37.9, 'Kansas': 37.1, 
                       'Kentucky': 39.1, 'Louisiana': 37.5, 
                       'Massachusetts': 39.6, 'Maryland': 39.1, 'Maine': 45.0, 
                       'Michigan': 39.9, 'Minnesota': 38.3, 'Missouri': 38.9, 
                       'Mississippi': 38.0, 'Montana': 40.1, 
                       'North_Carolina': 39.1, 'North_Dakota': 35.3, 
                       'Nebraska': 36.8, 'New_Hampshire': 43.1, 
                       'New_Jersey': 40.1, 'New_Mexico': 38.4, 'Nevada': 38.3,
                       'New_York': 39.2, 'Ohio': 39.5, 'Oklahoma': 36.9, 
                       'Oregon': 39.6, 'Pennsylvania': 40.8, 
                       'Rhode_Island': 40.1, 'South_Carolina': 39.9, 
                       'South_Dakota': 37.4, 'Tennessee': 39.0, 'Texas': 35.0,
                       'Utah':  31.3, 'Virginia': 38.6, 'Vermont': 43.0, 
                       'Washington': 37.8, 'Wisconsin': 39.8, 
                       'West_Virginia': 42.9, 'Wyoming': 38.4}

    return median_age_dict[state]


def import_contact_matrix(path, country, state):
    """Import contact matrix between age classes of a territory from a csv file 
    and store into an array.
    
    Original data from https://github.com/mobs-lab/mixing-patterns
    
    Parameters
    ----------
    path : str
        path where the csv file lies
    country : str
        name of the country
    state : str
        name of the region/state
        
    Returns
    -------
    - : np.array
        measures the average number of contacts for an individual of age i with 
        all of their contacts of age j.
    
    """

    lower_path = '/epivac_paper/data/'
    file_name = country + '_subnational_' + state + '_M_overall_contact_matrix_85'
    full_name = path + lower_path + file_name + '.csv'

    contact_df = pd.read_csv(full_name, header=None)

    return contact_df.values


def update_contact_matrix(contact, old_pop_a, new_pop_a):
    """Update contact matrices for every state from the 2005 version obtained
    in Mistry et al. (2020) to a more recent version (2019 census). 

    Updating proceeds by the methods developed in Arregui et al. (2018)
    https://doi.org/10.1371/journal.pcbi.1006638
    In particular, method 2, density correction is employed. End result comes
    from Eq. (5).
    
    Of course the method can be applied to every year.

    Parameters
    ----------
    contact : np.array
        measures the average number of contacts for an individual of age i with 
        all of their contacts of age j.
    old_pop_a : np.array
        population's distribution by age
    new_pop_a : np.array
        updated population's distribution by age

    Returns
    -------
    new_contact : np.array
        updated contact matrix

    """

    N_old = np.sum(old_pop_a)
    N_new = np.sum(new_pop_a)
    A = len(old_pop_a)
    new_contact = np.zeros((A, A), dtype=float)

    for i in range(A):

        for j in range(A):
            
            old_fraction = N_old / old_pop_a[j]
            new_fraction = new_pop_a[j] / N_new
            factor = old_fraction * new_fraction

            new_contact[i][j] = contact[i][j] * factor

    return new_contact


def import_initial_seeds(pop_a, path, country, state):
    """Import initial number of infected individuals for every age-class a.
    
    A reference base value is given arbitrarily for age-class between 5 and 18
    years old. The values for the rest of age-classes are given relative to the
    reference class. All of this following:
        https://www.cdc.gov/coronavirus/2019-ncov/covid-data/investigations-discovery/hospitalization-death-by-age.html

    Parameters
    ----------
    pop_a : np.array
        population's distribution by age
    path : str
        path where the csv file lies
    country : str
        name of the country
    state : str
        name of the region/state

    Returns
    -------
    seeds_a : np.array
        initial condition for the number of infected people by age class

    """

    seeds_a = np.zeros(len(pop_a))

    reference = 0.005 # Invent

    seeds_a[0:5] = 0.5 * reference * pop_a[0:5]
    seeds_a[5:18] = reference * pop_a[5:18]
    seeds_a[18:65] = 2.0 * reference * pop_a[18:65]
    seeds_a[65:84] = 1.0 * reference * pop_a[65:84]
    seeds_a[84] = 2.0 * reference * pop_a[84]

    return seeds_a


def import_immunized_fraction_by_state(pop_a, path, country, state):
    """Import the number of naturally immunized individuals by age for a given
    state at the start of the epidemic.
    
    These data are based on the reference Bajema et al. (2020)
    doi:10.1001/jamainternmed.2020.7976
    Data is extracted from the last of the seroprevalence surveys shown in the
    paper, September 7-September 24, 2020.

    Parameters
    ----------
    pop_a : np.array
        population's distribution by age
    path : str
        path where the csv file lies
    country : str
        name of the country
    state : str
        name of the region/state

    Returns
    -------
    - : np.array
        initial condition for the number of naturally immunized individuals for
        age-class

    """

    lower_path = '/epivac_paper/data/'
    file_name = 'corrected_state_immunized_fraction_by_age'
    full_name = path + lower_path + file_name + '.csv'

    recovered_df = pd.read_csv(full_name, sep=",", header=None, skiprows=0, 
                               index_col=0).T

    code = extract_code_from_state(state)

    immunized_by_age = np.zeros(85)
    
    immunized_by_age[0:18] = float(recovered_df[code][1]) * pop_a[0:18] / 100.0
    immunized_by_age[18:50] = float(recovered_df[code][2]) * pop_a[18:50] / 100.0
    immunized_by_age[50:65] = float(recovered_df[code][3]) * pop_a[50:65] / 100.0
    immunized_by_age[65:] = float(recovered_df[code][4]) * pop_a[65:] / 100.0

    return immunized_by_age


def import_death_rates_by_age(path, country, state):
    """Import infection fatality ratio (IFR) by age in the US.
    
    The IFR is extracted from table 2 in reference Verity et al. (2020):
        https://doi.org/10.1016/ S1473-3099(20)30243-7

    The CSV file contains the IFR (%) by age class

    Parameters
    ----------
    path : str
        path where the csv file lies
    country : str
        name of the country
    state : str
        name of the region/state
        
    Returns
    -------
    - : np.array
        fraction of deaths by age class 
    
    """

    lower_path = '/epivac_paper/data/'
    file_name = 'deaths_by_age'
    full_name = path + lower_path + file_name + '.csv'

    deaths_df = pd.read_csv(full_name, header=None, skiprows=1)
    
    deaths_array = np.zeros((3, 85), dtype=float)
    deaths_array[0] = deaths_df[:][1] / 100.0
    deaths_array[1] = deaths_df[:][2] / 100.0
    deaths_array[2] = deaths_df[:][3] / 100.0

    return deaths_array


def correct_vaccination_populations(vac_pop, pop_a):
    
    total_by_age = np.zeros(5)
    prop_factor = pop_a / np.sum(pop_a)
    
    total_by_age[0] = 0.0
    
    total_by_age[1] = np.sum(vac_pop[0][18:25]) + np.sum(vac_pop[1][18:25]) \
                    + np.sum(vac_pop[2][18:25]) + np.sum(vac_pop[3][18:25]) \
                    + np.sum(vac_pop[4][18:25])
    
    
    pop1825 = np.sum(pop_a[18:25])
    
    diff_1 = pop1825 - total_by_age[1]
    vac_pop[4][18:25] = vac_pop[4][18:25] \
                        + (diff_1 / (25 - 18)) * prop_factor[18:25]


    total_by_age[2] = np.sum(vac_pop[0][25:45]) + np.sum(vac_pop[1][25:45]) \
                    + np.sum(vac_pop[2][25:45]) + np.sum(vac_pop[3][25:45]) \
                    + np.sum(vac_pop[4][25:45])
    
    pop2545 = np.sum(pop_a[25:45])
    
    diff_2 = pop2545 - total_by_age[2]
    vac_pop[4][25:45] = vac_pop[4][25:45] \
                        + (diff_2 / (45 - 25)) * prop_factor[25:45]
    
    total_by_age[3] = np.sum(vac_pop[0][45:65]) + np.sum(vac_pop[1][45:65]) \
                    + np.sum(vac_pop[2][45:65]) + np.sum(vac_pop[3][45:65]) \
                    + np.sum(vac_pop[4][45:65])
    
    pop4565 = np.sum(pop_a[45:65])
    
    diff_3 = pop4565 - total_by_age[3]
    vac_pop[4][45:65] = vac_pop[4][45:65] \
                        + (diff_3 / (65 - 45)) * prop_factor[45:65]
                        
    total_by_age[4] = np.sum(vac_pop[0][65:]) + np.sum(vac_pop[1][65:]) \
                    + np.sum(vac_pop[2][65:]) + np.sum(vac_pop[3][65:]) \
                    + np.sum(vac_pop[4][65:])
    pop65 = np.sum(pop_a[65:])
    
    diff_4 = pop65 - total_by_age[4]
    vac_pop[4][65:] = vac_pop[4][65:] + (diff_4 / (85 - 65)) * prop_factor[65]

    return vac_pop


def import_vaccination_categories(pop_a, path, country, state, 
                                  counterfact=False):
    """Distribute vaccine attitude fractions of a given state by age
    groups.
    
    Original data of vaccination attitudes distinguishes various categories.
    See Lazer et al. (2021):
        Lazer, D., Ognyanova, K., Baum, M., Druckman, J., Green, J., Gitomer, 
        A., ... & Uslu, A. (2021). 
        The COVID States Project# 43: COVID-19 vaccine rates and attitudes 
        among Americans.
    
    This data is given at a national level in age groups, or at the level of
    states without disaggregation by ages. We want vaccination attitude 
    fractions distributed by age in a given state. Due to lack of finer data
    it is assumed that age-distributed attitudes are identical in every state.
    
    Dataframe age_nvaf_df contains vaccine acceptance categories in rows
    and age-groups in columns. 0: 18-24 y.o., 1: 25-44, 2: 45-64, 3: 65+ y.o.
    Elements of the dataframe are given as fraction normalized by total number 
    of people in each age class. This is the data given at a national level 
    that has then to be further redistributed into each state.
    
    Dataframe state_nvaf_df contains vaccine acceptance categories in rows
    and US states in columns. Elements of the dataframe are given as percentage
    of the, what I assumed, total adult population in every state. Every
    quantity here ought to be further distributed into age classes.
    
    To transform the data we proceed in the following way. Given a state,
    there is a share of its adult population in a certain vaccine acceptance
    category c. This share, or number of individuals N_c, has to be composed of
    individuals from different age classes in a certain proportion, N_ac, so 
    that N_c = f_a N_ac, where a is summed over all age classes. We want N_ac
    for every age-class a and every vaccine acceptance category c. 

    Parameters
    ----------
    pop_a : np.array
        population's distribution by age
    path : str
        path where the csv file lies
    country : str
        name of the country
    state : str
        name of the region/state
    
    Returns
    -------
    output : dict
        Contains the following keys:
        'age_nvaf' : array 
            global fractions of vaccine acceptance categories by age
        'state_nvaf': array
            percentage of vaccine acceptance categories by state
        'vacpop': array
            number of individuals in age-class a in vaccine acceptance category
            in a given US state
    
    """
    
    # Prepare output dictionary
    output = {}
    
    # Prepare full name of the files 
    lower_path = '/epivac_paper/data/'
    file_name1 = 'vaccination_attitude_fraction_by_age'
    full_name1 = path + lower_path + file_name1 + '.csv'
    file_name2 = 'vaccination_attitude_fraction_by_state'
    full_name2 = path + lower_path + file_name2 + '.csv'
    
    # Read CSV files as Pandas dataframe & store them into output dictionary
    age_nvaf_df = pd.read_csv(full_name1, sep=",", skiprows=1, names=range(4))
    output['age_nvaf'] = age_nvaf_df.values
    state_nvaf_df = pd.read_csv(full_name2, sep=",", skiprows=0, index_col=0).T
    output['state_nvaf'] = state_nvaf_df.values / 100.0
    
    prop = False
    if prop == True:
        
        if counterfact == True:
            code = 'National'
        else:
            code = extract_code_from_state(state)

        code = extract_code_from_state(state)
    
        vac_pop = np.zeros((5, 85))
        for att in range(5):
        
            vac_pop[att][0:18] = 0.0
            vac_pop[att][18:] = (state_nvaf_df[code][att] / 100.0) * pop_a[18:]

    else:
    
        # Get adult population in the state
        N_adults = np.sum(pop_a[18:])

        # Compute number of individuals of all ages in vaccination attitude cats
        age_global_pop_att = np.zeros(5)
        for att in range(5):
            summa = 0
            summa += age_nvaf_df[0][att] * np.sum(pop_a[18:25])
            summa += age_nvaf_df[1][att] * np.sum(pop_a[25:45])
            summa += age_nvaf_df[2][att] * np.sum(pop_a[45:65])
            summa += age_nvaf_df[3][att] * np.sum(pop_a[65:])
            age_global_pop_att[att] = summa

        if counterfact == True:
            code = 'National'
        else:
            code = extract_code_from_state(state)
    
        # Distribute into age classes
        vac_pop = np.zeros((5, 85))
        for att in range(5):
            state_pop_in_att = N_adults * (state_nvaf_df[code][att] / 100.0)
            pop_fraction = state_pop_in_att / age_global_pop_att[att]
            vac_pop[att,0:18] = 0.0
            vac_pop[att,18:25] = age_nvaf_df[0][att] * pop_a[18:25] * pop_fraction
            vac_pop[att,25:45] = age_nvaf_df[1][att] * pop_a[25:45] * pop_fraction
            vac_pop[att,45:65] = age_nvaf_df[2][att] * pop_a[45:65] * pop_fraction
            vac_pop[att,65:] = age_nvaf_df[3][att] * pop_a[65:] * pop_fraction
        
        correct_vaccination_populations(vac_pop, pop_a)

    output['vacpop'] = vac_pop

    return output
    

def extract_code_from_state(state):
    
    state_code_dict = {'Alaska': 'AK', 'Alabama': 'AL', 'Arkansas': 'AR', 
                       'Arizona': 'AZ', 'California': 'CA', 'Colorado': 'CO',
                       'Connecticut': 'CT', 'District_of_Columbia': 'DC',
                       'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA', 
                       'Hawaii': 'HI', 'Iowa': 'IA', 'Idaho': 'ID',
                       'Illinois': 'IL', 'Indiana': 'IN', 'Kansas': 'KS', 
                       'Kentucky': 'KY', 'Louisiana': 'LA', 
                       'Massachusetts': 'MA', 'Maryland': 'MD', 'Maine': 'ME', 
                       'Michigan': 'MI', 'Minnesota': 'MN', 'Missouri': 'MO', 
                       'Mississippi': 'MS', 'Montana': 'MT', 
                       'North_Carolina': 'NC', 'North_Dakota': 'ND', 
                       'Nebraska': 'NE', 'New_Hampshire': 'NH', 
                       'New_Jersey': 'NJ', 'New_Mexico': 'NM', 'Nevada': 'NV',
                       'New_York': 'NY', 'Ohio': 'OH', 'Oklahoma': 'OK', 
                       'Oregon': 'OR', 'Pennsylvania': 'PA', 
                       'Rhode_Island': 'RI', 'South_Carolina': 'SC', 
                       'South_Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX',
                       'Utah':  'UT', 'Virginia': 'VA', 'Vermont': 'VT', 
                       'Washington': 'WA', 'Wisconsin': 'WI', 
                       'West_Virginia': 'WV', 'Wyoming': 'WY'}

    return state_code_dict[state]


def extract_state_from_code(code):

    code_state_dict = {'AK': 'Alaska', 'AL': 'Alabama', 'AR': 'Arkansas', 
                       'AZ': 'Arizona','CA': 'California', 'CO': 'Colorado',
                       'CT': 'Connecticut', 'DC': 'District_of_Columbia',
                       'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia', 
                       'HI': 'Hawaii', 'IA': 'Iowa', 'ID': 'Idaho',
                       'IL': 'Illinois', 'IN': 'Indiana', 'KS': 'Kansas', 
                       'KY': 'Kentucky', 'LA': 'Louisiana', 
                       'MA': 'Massachusetts', 'MD': 'Maryland', 'ME': 'Maine', 
                       'MI': 'Michigan', 'MN': 'Minnesota', 'MO': 'Missouri', 
                       'MS': 'Mississippi', 'MT': 'Montana', 
                       'NC': 'North_Carolina', 'ND': 'North_Dakota', 
                       'NE': 'Nebraska', 'NH': 'New_Hampshire', 
                       'NJ': 'New_Jersey', 'NM': 'New_Mexico', 'NV': 'Nevada',
                       'NY': 'New_York', 'OH': 'Ohio', 'OK': 'Oklahoma', 
                       'OR': 'Oregon', 'PA': 'Pennsylvania', 
                       'RI': 'Rhode_Island', 'SC': 'South_Carolina', 
                       'SD': 'South_Dakota', 'TN': 'Tennessee', 'TX': 'Texas',
                       'UT': 'Utah',  'VA': 'Virginia', 'VT': 'Vermont', 
                       'WA': 'Washington', 'WI': 'Wisconsin', 
                       'WV': 'West_Virginia', 'WY': 'Wyoming'}
    
    return code_state_dict[code]


def extract_population_data(path, country, state, update=True, zeroprev=False,
                            counterfact=False):
    """Extract all needed population-related data. 
    
    Population-related data includes:
        - Age distribution. 
        - Contact matrix. 
        - Seeds in the 1st outbreak.
        - Naturally immunized before the 1st outbreak.
        - Already vaccinated before 1st outbreak. Region-dependent data.
        - Different attitude fractions to be vaccinated. Region-dependent data.

    Parameters
    ----------
    path : str
        general path where the data lies
    country : str
        name of the country
    state : str
        name of the region/state
    update : bool (optional)
        If True, 2019 updated population data is used. True by default.
    zeroprev : bool (optional)
        If True, no seroprevalence real data is used. No naturally immunized
        as initial condition. False by default.
        
    Returns
    -------
    data : dict
        Contains the following keys:
            country : str
            state : str
            path : str
            pop_a : np.array
            contact : np.array
            immunized : np.array
            seeds : np.array
            vaccination : np.array
            deaths : np.array

    """

    data = {}
    
    data['country'] = country
    data['state'] = state
    data['path'] = path
    
    if update is False:
        pop_a = import_age_distribution(path, country, state)
        data['pop_age'] = pop_a
        
        contact = import_contact_matrix(path, country, state)
        data['contact'] = contact
        
    else:
        old_pop_a = import_age_distribution(path, country, state)
        pop_a = import_updated_age_distribution(path, country, state)
        data['pop_age'] = pop_a

        contact = import_contact_matrix(path, country, state)
        data['contact'] = update_contact_matrix(contact, old_pop_a, pop_a)

    if zeroprev is True:
        data['immunized'] = 0.0 * pop_a
        
    else:
        data['immunized'] = import_immunized_fraction_by_state(pop_a, path, 
                                                               country, state)

    data['seeds'] = import_initial_seeds(pop_a, path, country, state)
    
    data['vaccination'] = import_vaccination_categories(pop_a, path, country, 
                                                        state, counterfact)
    
    data['deaths'] = import_death_rates_by_age(path, country, state)

    return data


def feed_the_model(path, country, state, update=True, zeroprev=False, 
                   counterfact=False):
    """Obtain all the necessary data and parameters to initialize and make
    work the epidemic models.
    
    All parameters are stored in a common dictionary.
    
    Parameters
    ----------
    path : str
        general path where the data lies
    country : str
        name of the country
    state : str
        name of the region/state
    update : bool (optional)
        If True, 2019 updated population data is used. True by default.
    zeroprev : bool (optional)
        If True, no seroprevalence real data is used. No naturally immunized
        as initial condition. False by default.
    
    Returns
    -------
    mpars : dict
        Contains the following keys:
        R0 : float
        R0_P2 : float
        gamma : float
        path : str
        country : str
        state : str
        pop_age : np.array
        contact : np.array
        vaccination : dict
        immunized : np.array
        seeds : np.array
        deaths : np.array
        t_start : float
        t_vac : float

    """

    dpars = extract_population_data(path, country, state, update, zeroprev,
                                    counterfact)

    # Set disease & vaccination campaign parameters
    R0 = 1.5 # Phase I reproductive number
    R0_P2 = 3.0 # Phase II reproductive number
    gamma = 1.0 / 4.5 # Since this is not a SEIR, could be more appropriate
    t_start = 0
    t_vac = 150 - t_start # Arbitrary length (in days) of vaccination campaign

    mpars = {}
    mpars['R0'] = R0
    mpars['R0_P2'] = R0_P2
    mpars['second_wave'] = True
    mpars['gamma'] = gamma
    mpars['chi_ua'] = 0.56 # Some ref. should be given to justify this value
    mpars['path'] = dpars['path']
    mpars['country'] = dpars['country']
    mpars['state'] = dpars['state']
    mpars['pop_age'] = dpars['pop_age']
    mpars['contact'] = dpars['contact']
    mpars['vaccination'] = dpars['vaccination']
    mpars['immunized'] = dpars['immunized']
    mpars['seeds'] = dpars['seeds']
    mpars['deaths'] = dpars['deaths']
    mpars['t_start'] = t_start
    mpars['t_vac'] = t_vac
    
    print('Data loaded')

    return mpars