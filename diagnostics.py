# -*- coding: utf-8 -*-
import simulate
import calibrate_begs_pers_log as Para
#import calibrate_begs_iid as Para
import numpy as np
import pandas as pd
Gamma,Y, Shocks, y= {},{},{},{}
N=100
T=20




indx_y=Para.indx_y
indx_Y=Para.indx_Y
indx_Gamma=Para.indx_Gamma
Gamma[0] = np.zeros((N,Para.nz)) 
Gamma,Y,Shocks,y=simulate.simulate(Para,Gamma,Y,T)

 
mu,output,g,assets,bar_g,simulation_data={},{},{},{},{},{}
for t in range(T-1):
    if np.shape(y[t])[1]<10:
        y[t]=np.hstack((y[t][:,0:1],np.atleast_2d(Shocks[0]).reshape(N,1),y[t][:,1:]))
    
    output[t]=np.atleast_2d(y[t][:,indx_y['l']]*np.exp(y[t][:,indx_y['e']])).reshape(N,1)
    mu[t]=np.exp(y[t][:,indx_y['logm']])*(y[t][:,indx_y['muhat']])
    g[t]=(1-mu[t]*(1+Para.gamma))*((y[t][:,indx_y['c']])**(-Para.sigma))
    bar_g[t]=np.mean(g[t])    
    g[t]=g[t]/bar_g[t]    
    g[t]=np.atleast_2d(g[t]).reshape(N,1)
    if t==0:
       assets[t]=y[t][:,indx_y['x_']]*(1/np.exp(Gamma[t][:,indx_Gamma['m_']]))
    else:
       assets[t]=y[t][:,indx_y['x_']]*(Y[t-1][indx_Y['alpha2']]/Gamma[t][:,indx_Gamma['m_']])    
       

    assets[t]=np.atleast_2d(assets[t]).reshape(N,1)    
    simulation_data[t]=np.hstack((output[t],g[t],assets[t],y[t],np.atleast_2d(Shocks[t]).reshape(N,1)))

panel_data=pd.Panel(simulation_data,minor_axis=['y','g','b','logm','muhat','e','c','l','rho1_','rho2','phi','x_','kappa_','shocks'])

panel_data.to_pickle('simulation_data.dat')

def get_quantiles(var):
    var_data=panel_data.minor_xs(var)
    low_q=var_data.quantile(.25,axis=0)
    m_q=var_data.quantile(.5,axis=0)
    high_q=var_data.quantile(.75,axis=0)    
    plot(array([low_q,m_q,high_q]).T)
    return low_q,m_q,high_q


def get_corr(var_1,var_2):
    corr_data= map(lambda t: np.corrcoef(panel_data.minor_xs(var_1)[t],panel_data.minor_xs(var_2)[t])[0,1],range(T-1))
    plot(corr_data)
    return corr_data


def plot_agg_var(var): 
    plot(map(lambda t: Y[t][indx_Y[var]],range(T-1)))
    
