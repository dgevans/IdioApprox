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

listofdatafiles=['initialization',
'irf_high_tfp_no_idiosyncratic_shocks',
'irf_high_tfp_with_idiosyncratic_shocks',
'irf_low_tfp_no_idiosyncratic_shocks',
'irf_low_tfp_with_idiosyncratic_shocks',
'long_drift_high_shocks',
'long_drift_high_tfp_no_idiosyncratic_shocks',
'long_drift_low_shocks',
'long_drift_no_shocks',
'long_sample_no_agg_shock',
'long_sample_with_agg_shock']


for name in listofdatafiles:
    name_of_file='data'+'_'+name+'.pickle'    
    data = pickle.load( open( name_of_file, "rb" ) )    
    copy_command='cp *.png /home/anmol/IdioApprox/Graphs/'+ name+ '/'
    
    Gamma,Y,Shocks,y=data
    indx_y,indx_Y,indx_Gamma=Para.indx_y,Para.indx_Y,Para.indx_Gamma
    T=len(Shocks)
    N=np.shape(Gamma[0])[0]
    mu,output,g,assets,bar_g,simulation_data={},{},{},{},{},{}
    for t in range(T-1):
        output[t]=np.atleast_2d(y[t][:,indx_y['l']]*np.exp(y[t][:,indx_y['wages']])).reshape(N,1)
        mu[t]=np.exp(y[t][:,indx_y['logm']])*(y[t][:,indx_y['muhat']])
        #g[t]=(1-mu[t]*(1+Para.gamma))*((y[t][:,indx_y['c']])**(-Para.sigma))
        #bar_g[t]=np.mean(g[t])    
        #g[t]=g[t]/bar_g[t]    
        #g[t]=np.atleast_2d(g[t]).reshape(N,1)
        #if t==0:
        #   assets[t]=y[t][:,indx_y['x_']]*(1/np.exp(Gamma[t][:,indx_Gamma['m_']]))
        #else:
        #   assets[t]=y[t][:,indx_y['x_']]*(Y[t-1][indx_Y['alpha2']]/np.exp(Gamma[t][:,indx_Gamma['m_']]))    
           
    
        #assets[t]=np.atleast_2d(assets[t]).reshape(N,1)    
        #simulation_data[t]=np.hstack((output[t],g[t],assets[t],np.atleast_2d(mu[t]).reshape(N,1),y[t],np.atleast_2d(Shocks[t]).reshape(N,1)))
        #simulation_data[t]=np.hstack((output[t],np.atleast_2d(mu[t]).reshape(N,1),y[t],np.atleast_2d(Shocks[t]).reshape(N,1)))
        simulation_data[t]=np.hstack((output[t],np.atleast_2d(mu[t]).reshape(N,1),y[t]))
        simulation_data[t]=np.hstack((output[t],np.atleast_2d(mu[t]).reshape(N,1),y[t]))
        panel_data=pd.Panel(simulation_data,minor_axis=['y','mu','logm','muhat','e','c' ,'l','rho1_','rho2','phi','wages','UcP','a','x_','kappa_'])
    
    #panel_data.to_pickle('simulation_data.dat')
    
    def get_quantiles(var):
        var_data=panel_data.minor_xs(var)
        low_q=var_data.quantile(.10,axis=0)
        m_q=var_data.quantile(.5,axis=0)
        high_q=var_data.quantile(.9,axis=0)    
        plt.plot(np.array([low_q,m_q,high_q]).T)
        return low_q,m_q,high_q
    
    
    def get_cov(var_1,var_2):
        cov_data= map(lambda t: np.cov(panel_data.minor_xs(var_1)[t],panel_data.minor_xs(var_2)[t])[0,1],range(T-1))
        
        return cov_data
    
    
    
    def get_scatter(var_1,var_2,t):
        plt.scatter(panel_data.minor_xs(var_1)[t],panel_data.minor_xs(var_2)[t])
        
        
    
    
    def plot_agg_var(var): 
        time_series_data=map(lambda t: Y[t][indx_Y[var]],range(T-1))    
        plt.plot(time_series_data,'k')
        return time_series_data
        
    
    
    
    low_q_c,m_q_c,high_q_c=get_quantiles('c')
    
    
    low_q_y,m_q_y,high_q_y=get_quantiles('y')
    
    low_q_l,m_q_l,high_q_l=get_quantiles('l')
    
    low_q_a,m_q_a,high_q_a=get_quantiles('a')
    
    
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
    plot_agg_var('taxes')
    plt.title('Tax rates')
    plt.xlabel('t')
    plt.ylabel(r'$\tau$')
    #figure = plt.gcf() # get current figure
    #figure.set_size_inches(8, 6)
    # when saving, specify the DPI
    plt.savefig('taxes.png',dpi=300)
    
    
    
    plt.figure()
    plot_agg_var('T')
    plt.title('Transfers')
    plt.xlabel('t')
    plt.ylabel(r'$T$')
    #figure = plt.gcf() # get current figure
    #figure.set_size_inches(8, 6)
    # when saving, specify the DPI
    plt.savefig('Transfers.png',dpi=300)
    
    
    plt.figure()
    plt.plot(sum(panel_data.minor_xs('a')),'k')
    plt.title('Debt')
    plt.xlabel('t')
    plt.ylabel(r'$-B_t$')
    #figure = plt.gcf() # get current figure
    #figure.set_size_inches(8, 6)
    # when saving, specify the DPI
    plt.savefig('Debt.png',dpi=300)
    
    
    
    cov_data_c_y=get_cov('c','y')
    cov_data_a_y=get_cov('a','y')
    cov_data_a_c=get_cov('a','c')
    cov_data_l_c=get_cov('l','c')
    
    
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
    
    
    plt.savefig('covariances.png',dpi=300)
    
    
    
    
    cov_data_c_c=get_cov('c','c')
    cov_data_a_a=get_cov('a','a')
    cov_data_l_l=get_cov('l','l')
    cov_data_y_y=get_cov('y','y')
    
    
    f,((ax1,ax2),(ax3,ax4)) =plt.subplots(2,2,sharex='col')
    lines_cov_c_c=ax1.plot(cov_data_c_c)
    lines_cov_a_a=ax2.plot(cov_data_a_a)
    lines_cov_l_l=ax3.plot(cov_data_l_l)
    lines_cov_y_y=ax4.plot(cov_data_y_y)
    
    plt.setp(lines_cov_c_c,color='k')
    plt.setp(lines_cov_a_a,color='k')
    plt.setp(lines_cov_l_l,color='k')
    plt.setp(lines_cov_y_y,color='k')
    
    
    ax1.set_title(r'c')
    ax2.set_title(r'a')
    ax3.set_title(r'l')
    ax4.set_title(r'y')
    
    
    plt.savefig('variances.png',dpi=300)
    
    os.system(copy_command)