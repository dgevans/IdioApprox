# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 19:49:53 2014

@author: dgevans
"""

import calibrate_begs_id_nu_ces as Para
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cPickle as pickle
import os as os
listofdatafiles=[
'initialization',
'irf_high_tfp_no_idiosyncratic_shocks',
'irf_high_tfp_no_idiosyncratic_shocks',
'irf_high_tfp_with_idiosyncratic_shocks',
'irf_low_tfp_no_idiosyncratic_shocks',
'irf_low_tfp_with_idiosyncratic_shocks',
'long_drift_high_shocks',
'long_drift_high_tfp_no_idiosyncratic_shocks',
'long_drift_low_tfp_no_idiosyncratic_shocks',
'long_drift_low_shocks',
'long_drift_no_shocks',
'long_sample_no_agg_shock',
'long_sample_with_agg_shock'
]


for name in listofdatafiles:
    print name
    name_of_file='/home/anmol/IdioApprox/plot_'+'data'+'_'+name+'.pickle'    
    data = pickle.load( open( name_of_file, "rb" ) )    
    copy_command='cp *.png /home/anmol/IdioApprox/Graphs/'+ name+ '/'
    Y,low_q_c,m_q_c,high_q_c,low_q_y,m_q_y,high_q_y,low_q_l,m_q_l,high_q_l,low_q_a,m_q_a,high_q_a,cov_data_c_y,cov_data_a_y,cov_data_a_c,cov_data_l_c,var_data_c,var_data_a,var_data_l,var_data_y,debt=data
        
    
    plt.close('all')
    f,((ax1,ax2),(ax3,ax4)) =plt.subplots(2,2,sharex='col')
    lines_c=ax1.plot( np.array([low_q_c,m_q_c,high_q_c]).T)
    lines_y=ax2.plot( np.array([low_q_y,m_q_y,high_q_y]).T)
    lines_l=ax3.plot( np.array([low_q_l,m_q_l,high_q_l]).T)
    lines_a=ax4.plot( np.array([low_q_a,m_q_a,high_q_a]).T)
    
    plt.setp(lines_c[0],color='k',ls='--')
    plt.setp(lines_c[1],color='k')
    plt.setp(lines_c[2],color='k',ls='--')
    
    
    plt.setp(lines_y[0],color='k',ls='--')
    plt.setp(lines_y[1],color='k')
    plt.setp(lines_y[2],color='k',ls='--')
    
    
    plt.setp(lines_l[0],color='k',ls='--')
    plt.setp(lines_l[1],color='k')
    plt.setp(lines_l[2],color='k',ls='--')
    
    
    
    plt.setp(lines_a[0],color='k',ls='--')
    plt.setp(lines_a[1],color='k')
    plt.setp(lines_a[2],color='k',ls='--')
    
    
    
    
    ax1.set_title(r'consumption')
    
    ax2.set_title(r'output')
    
    ax3.set_title(r'labor')
    
    ax4.set_title(r'assets')
    
    
    plt.savefig('quantiles.png',dpi=300)
    
    
    
    
    plt.figure()
    plt.plot(map(lambda t: Y[t][Para.indx_Y['taxes']],range(len(Y)-1)),'k')
    plt.title('Tax rates')
    plt.xlabel('t')
    plt.ylabel(r'$\tau$')
    #figure = plt.gcf() # get current figure
    #figure.set_size_inches(8, 6)
    # when saving, specify the DPI
    plt.savefig('taxes.png',dpi=300)
    
    
    
    plt.figure()
    plt.plot(map(lambda t: Y[t][Para.indx_Y['T']],range(len(Y)-1)),'k')
    plt.title('Transfers')
    plt.xlabel('t')
    plt.ylabel(r'$T$')
    #figure = plt.gcf() # get current figure
    #figure.set_size_inches(8, 6)
    # when saving, specify the DPI
    plt.savefig('Transfers.png',dpi=300)
    
    
    plt.figure()
    plt.plot(debt,'k')
    plt.title('Debt')
    plt.xlabel('t')
    plt.ylabel(r'$-B_t$')
    #figure = plt.gcf() # get current figure
    #figure.set_size_inches(8, 6)
    # when saving, specify the DPI
    plt.savefig('Debt.png',dpi=300)
    
    
    
    
    f,((ax1,ax2),(ax3,ax4)) =plt.subplots(2,2,sharex='col')
    lines_cov_c_y=ax1.plot(cov_data_c_y)
    lines_cov_a_y=ax2.plot(cov_data_a_y)
    lines_cov_a_c=ax3.plot(cov_data_a_c)
    lines_cov_l_c=ax4.plot(cov_data_l_c)
    
    plt.setp(lines_cov_c_y,color='k')
    plt.setp(lines_cov_a_y,color='k')
    plt.setp(lines_cov_a_c,color='k')
    plt.setp(lines_cov_l_c,color='k')
    
    
    ax1.set_title(r'c,y')
    ax2.set_title(r'a,y')
    ax3.set_title(r'a,c')
    ax4.set_title(r'l,c')
    
    
    plt.savefig('correlations.png',dpi=300)
    
    
    
    
    
    
    f,((ax1,ax2),(ax3,ax4)) =plt.subplots(2,2,sharex='col')
    lines_cov_c_c=ax1.plot(var_data_c)
    lines_cov_a_a=ax2.plot(var_data_a)
    lines_cov_l_l=ax3.plot(var_data_l)
    lines_cov_y_y=ax4.plot(var_data_y)
    
    plt.setp(lines_cov_c_c,color='k')
    plt.setp(lines_cov_a_a,color='k')
    plt.setp(lines_cov_l_l,color='k')
    plt.setp(lines_cov_y_y,color='k')
    
    
    ax1.set_title(r'c')
    ax2.set_title(r'a')
    ax3.set_title(r'l')
    ax4.set_title(r'y')
    
    
    plt.savefig('coeff_variation.png',dpi=300)
    
    os.system(copy_command)