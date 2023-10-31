#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 22:48:27 2021

@author: alfonso

Just control prints

"""


def control_prints(data):
    
    res_fo, res_so, res_fe = data
    
    nva_ss_fo_b = round(res_fo['nva_ss'], 8)
    sus_ss_fo_b = round(res_fo['sus_ss'], 8)
    sus_ss_so_b = round(res_so['sus_ss'], 8)
    ove_ss_fo_b = round(res_fo['ove'], 8)
    ove_ss_so_b = round(res_so['ove'], 8)
    pre_ss_fo_b = round(res_fo['pre_ss'], 8)
    pre_ss_so_b = round(res_so['pre_ss'], 8)
    pre_ss_fe_b = round(res_fe['pre_ss'], 8)
    dea_ss_fo_b = round(res_fo['dea'], 8)
    dea_ss_so_b = round(res_so['dea'], 8)
    dea_ss_fe_b = round(res_fe['dea'], 8)
    imm_ss_fo_b = round(res_fo['imm_ss'], 8)
    imm_ss_so_b = round(res_so['imm_ss'], 8)
    imm_ss_fe_b = round(res_fe['imm_ss'], 8)

    print('Non vaccinated: {0}'.format(nva_ss_fo_b))
    print('Remaining susceptible after 1st outbreak: {0}'.format(sus_ss_fo_b))
    print('Remanining susceptible after 2nd outbreak: {0}'.format(sus_ss_so_b))
    print('Overshoot: {0}'.format(ove_ss_fo_b))
    print('Overshoot: {0}'.format(ove_ss_so_b))
    print('Attacked in 1st outbreak: {0}'.format(pre_ss_fo_b))
    print('Attacked in 2nd outbreak: {0}'.format(pre_ss_so_b))
    print('Attacked in full epidemic: {0}'.format(pre_ss_fe_b))
    print('Deaths in 1st outbreak:{0}'.format(dea_ss_fo_b))
    print('Deaths in 2nd outbreak:{0}'.format(dea_ss_so_b))
    print('Deaths in full epidemic:{0}'.format(dea_ss_fe_b))
    print('Immunized in 1st outbreak: {0}'.format(imm_ss_fo_b))
    print('Immunized in 2nd outbreak: {0}'.format(imm_ss_so_b))
    print('Immunized in full epidemic: {0}'.format(imm_ss_fe_b))
    
    diff_sus_ss_fo = sus_ss_fo_b - sus_ss_fo_e
    diff_pre_ss_fo = pre_ss_fo_b - pre_ss_fo_e
    diff_pre_ss_so = pre_ss_so_b - pre_ss_so_e
    diff_pre_ss_fe = pre_ss_fe_b - pre_ss_fe_e
    diff_dea_ss_fo = dea_ss_fo_b - dea_ss_fo_e
    diff_dea_ss_so = dea_ss_so_b - dea_ss_so_e
    diff_dea_ss_fe = dea_ss_fe_b - dea_ss_fe_e
    diff_imm_ss_fo = imm_ss_fo_b - imm_ss_fo_e
    diff_imm_ss_so = imm_ss_so_b - imm_ss_so_e
    diff_imm_ss_fe = imm_ss_fe_b - imm_ss_fe_e
    
    print('Models comparison')
    print('Susceptible diff in 1st outbreak: {0}'.format(diff_sus_ss_fo))
    print('Attacked diff in 1st outbreak: {0}'.format(diff_pre_ss_fo))
    print('Attacked diff in 2nd outbreak: {0}'.format(diff_pre_ss_so))
    print('Attacked diff in full epidemic: {0}'.format(diff_pre_ss_fe))
    print('Death diff in 1st outbreak: {0}'.format(diff_dea_ss_fo))
    print('Death diff in 2nd outbreak: {0}'.format(diff_dea_ss_so))
    print('Death diff in full epidemic: {0}'.format(diff_dea_ss_fe))
    print('Immunized diff in 1st outbreak: {0}'.format(diff_imm_ss_fo))
    print('Immunized diff in 2nd outbreak: {0}'.format(diff_imm_ss_so))
    print('Immunized diff in full epidemic: {0}'.format(diff_imm_ss_fe))

