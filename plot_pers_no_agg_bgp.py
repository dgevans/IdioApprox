# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 19:49:53 2014

@author: dgevans
"""

import simulate
import calibrate_begs_id_nu_bgp as Para
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
N=500
T=100
Gamma,Y,Shocks,y = {},{},{},{}

Gamma[0] = np.zeros((N,3)) #initialize 100 agents at m = 1 for testing purposes
Gamma[0][:,0] = np.zeros(N)


v = simulate.v
v.execute('import calibrate_begs_id_nu_bgp as Para')
v.execute('import approximate_begs as approximate')
v.execute('approximate.calibrate(Para)')
simulate.simulate(Para,Gamma,Y,Shocks,y,T)

indx_y,indx_Y,indx_Gamma=Para.indx_y,Para.indx_Y,Para.indx_Gamma

mu,output,g,assets,bar_g,simulation_data={},{},{},{},{},{}
for t in range(T-1):
    if np.shape(y[t])[1]<10:
        y[t]=np.hstack((y[t][:,0:1],np.atleast_2d(Shocks[0]).reshape(N,1),y[t][:,1:]))
    
    output[t]=np.atleast_2d(y[t][:,indx_y['l']]*np.exp(y[t][:,indx_y['e']])).reshape(N,1)
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
    simulation_data[t]=np.hstack((output[t],np.atleast_2d(mu[t]).reshape(N,1),y[t],np.atleast_2d(Shocks[t]).reshape(N,1)))
    simulation_data[t]=np.hstack((output[t],np.atleast_2d(mu[t]).reshape(N,1),y[t],np.atleast_2d(Shocks[t]).reshape(N,1)))
    panel_data=pd.Panel(simulation_data,minor_axis=['y','mu','logm','muhat','e','c','l','rho1_','rho2','phi','x_','kappa_','shocks'])

#panel_data.to_pickle('simulation_data.dat')

def get_quantiles(var):
    var_data=panel_data.minor_xs(var)
    low_q=var_data.quantile(.25,axis=0)
    m_q=var_data.quantile(.5,axis=0)
    high_q=var_data.quantile(.75,axis=0)    
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

cov_data_c_y=get_cov('c','y')
plt.close('all')
f,((ax1,ax2),(ax3,ax4)) =plt.subplots(2,2,sharex='col')
lines_c=ax1.plot( np.array([low_q_c,m_q_c,high_q_c]).T)
lines_y=ax2.plot( np.array([low_q_y,m_q_y,high_q_y]).T)
lines_l=ax3.plot( np.array([low_q_l,m_q_l,high_q_l]).T)
lines_cov=ax4.plot(cov_data_c_y)

plt.setp(lines_c[0],color='k',ls='--')
plt.setp(lines_c[1],color='k')
plt.setp(lines_c[2],color='k',ls='--')


plt.setp(lines_y[0],color='k',ls='--')
plt.setp(lines_y[1],color='k')
plt.setp(lines_y[2],color='k',ls='--')


plt.setp(lines_l[0],color='k',ls='--')
plt.setp(lines_l[1],color='k')
plt.setp(lines_l[2],color='k',ls='--')


plt.setp(lines_cov[0],color='k')



ax1.set_title(r'consumption')

ax2.set_title(r'output')

ax3.set_title(r'output')

ax4.set_title(r'cov(c,y)')

plt.savefig('quantiles_pers_no_agg_shocks.png',dpi=300)




plt.figure()
plot_agg_var('taxes')
plt.title('Tax rates')
plt.xlabel('t')
plt.ylabel(r'$\tau$')
#figure = plt.gcf() # get current figure
#figure.set_size_inches(8, 6)
# when saving, specify the DPI
plt.savefig('taxes_pers_no_agg_shocks.png',dpi=300)


