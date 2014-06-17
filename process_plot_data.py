# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 19:49:53 2014

@author: dgevans
"""




import calibrate_begs_id_nu_ces as Para
import numpy as np
import pandas as pd
import cPickle as pickle

global T,truncation, panel_data

truncation=.99

def get_panel_data(data):
        
    Y,y=data
    indx_y,indx_Y,indx_Gamma=Para.indx_y,Para.indx_Y,Para.indx_Gamma
    
    T=len(Y)
    
    N=np.shape(y[0])[0]
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
    return panel_data

    #panel_data.to_pickle('simulation_data.dat')
    
def get_quantiles(var,panel_data):
        var_data=panel_data.minor_xs(var)
        low_q=var_data.quantile(.10,axis=0)
        m_q=var_data.quantile(.5,axis=0)
        high_q=var_data.quantile(.9,axis=0)            
        return low_q,m_q,high_q
    
    
def get_cov(var_1,var_2,panel_data):
        T=len(panel_data)            

        cov_data= map(lambda t: np.corrcoef(panel_data.minor_xs(var_1)[t][panel_data.minor_xs(var_1)[t]<panel_data.minor_xs(var_1)[t].quantile(truncation)],panel_data.minor_xs(var_2)[t][panel_data.minor_xs(var_2)[t]<panel_data.minor_xs(var_2)[t].quantile(truncation)])[0,1],range(T-1))
        
        return cov_data
    

    
def get_var(variable,panel_data):
        T=len(panel_data)

        var_data= map(lambda t:  np.std(panel_data.minor_xs(variable)[t][panel_data.minor_xs(variable)[t]<panel_data.minor_xs(variable)[t].quantile(truncation)])/(np.mean(panel_data.minor_xs(variable)[t][panel_data.minor_xs(variable)[t]<panel_data.minor_xs(variable)[t].quantile(truncation)])),range(T-1))
        
        return var_data    
    
        
    
        
    
def save_plot_data(data,nameoffile):
    
    
    Y,y=data
    
    panel_data=get_panel_data(data)
    
    
    low_q_c,m_q_c,high_q_c=get_quantiles('c',panel_data)
    low_q_y,m_q_y,high_q_y=get_quantiles('y',panel_data)
    low_q_l,m_q_l,high_q_l=get_quantiles('l',panel_data)
    low_q_a,m_q_a,high_q_a=get_quantiles('a',panel_data)
        
    
    
    cov_data_c_y=get_cov('c','y',panel_data)
    cov_data_a_y=get_cov('a','y',panel_data)
    cov_data_a_c=get_cov('a','c',panel_data)
    cov_data_l_c=get_cov('l','c',panel_data)
    

    
    var_data_c=get_var('c',panel_data)
    var_data_a=get_var('a',panel_data)
    var_data_l=get_var('l',panel_data)
    var_data_y=get_var('y',panel_data)
    
    
    store_data=Y,low_q_c,m_q_c,high_q_c,low_q_y,m_q_y,high_q_y,low_q_l,m_q_l,high_q_l,low_q_a,m_q_a,high_q_a,cov_data_c_y,cov_data_a_y,cov_data_a_c,cov_data_l_c,var_data_c,var_data_a,var_data_l,var_data_y
    
    with open(nameoffile, 'wb') as f:
        pickle.dump(store_data, f)
