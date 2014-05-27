# -*- coding: utf-8 -*-
import simulate
#import calibrate_begs_pers_log as Para
import calibrate_begs_iid as Para
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Gamma,Y, Shocks, y= {},{},{},{}
N=10000
T=400

#
v = simulate.v
v.execute('import calibrate_begs_iid as Para')
v.execute('import approximate_parallel as approximate')
v.execute('approximate.calibrate(Para)')



indx_y=Para.indx_y
indx_Y=Para.indx_Y
indx_Gamma=Para.indx_Gamma
Gamma[0] = np.zeros((N,Para.nz)) 
simulate.simulate(Para,Gamma,Y,Shocks,y,T)

 
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
       assets[t]=y[t][:,indx_y['x_']]*(Y[t-1][indx_Y['alpha2']]/np.exp(Gamma[t][:,indx_Gamma['m_']]))    
       

    assets[t]=np.atleast_2d(assets[t]).reshape(N,1)    
    simulation_data[t]=np.hstack((output[t],g[t],assets[t],np.atleast_2d(mu[t]).reshape(N,1),y[t],np.atleast_2d(Shocks[t]).reshape(N,1)))

panel_data=pd.Panel(simulation_data,minor_axis=['y','g','b','mu','logm','muhat','e','c','l','rho1_','rho2','phi','x_','kappa_','shocks'])

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
    plt.plot(time_series_data)
    return time_series_data
    






plt.subplot(2, 2, 1)
low_q,m_q,high_q=get_quantiles('c')

plt.plot(np.array([low_q,m_q,high_q]).T)
plt.title('consumption')

plt.subplot(2, 2, 2)
low_q,m_q,high_q=get_quantiles('y')

plt.plot(np.array([low_q,m_q,high_q]).T)
plt.title('output')


plt.subplot(2, 2, 3)
low_q,m_q,high_q=get_quantiles('l')

plt.plot(np.array([low_q,m_q,high_q]).T)
plt.title('output')

plt.subplot(2, 2, 4)
cov_data_c_y=get_cov('c','y')
plt.plot(cov_data_c_y)

plt.figure()
plot_agg_var('taxes')

Y_iid = Y
panel_data_iid = panel_data


N = 1000
import calibrate_begs_pers_log as Para
v.execute('import calibrate_begs_pers_log as Para')
v.execute('import approximate_parallel as approximate')
v.execute('approximate.calibrate(Para)')


Gamma,Y, Shocks, y= {},{},{},{}
indx_y=Para.indx_y
indx_Y=Para.indx_Y
indx_Gamma=Para.indx_Gamma
Gamma[0] = np.zeros((N,Para.nz)) 
simulate.simulate(Para,Gamma,Y,Shocks,y,T)

 
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
       assets[t]=y[t][:,indx_y['x_']]*(Y[t-1][indx_Y['alpha2']]/np.exp(Gamma[t][:,indx_Gamma['m_']]))    
       

    assets[t]=np.atleast_2d(assets[t]).reshape(N,1)    
    simulation_data[t]=np.hstack((output[t],g[t],assets[t],np.atleast_2d(mu[t]).reshape(N,1),y[t],np.atleast_2d(Shocks[t]).reshape(N,1)))

panel_data=pd.Panel(simulation_data,minor_axis=['y','g','b','mu','logm','muhat','e','c','l','rho1_','rho2','phi','x_','kappa_','shocks'])

plt.figure()
plt.subplot(2, 2, 1)
low_q,m_q,high_q=get_quantiles('c')

plt.plot(np.array([low_q,m_q,high_q]).T)
plt.title('consumption')

plt.subplot(2, 2, 2)
low_q,m_q,high_q=get_quantiles('y')

plt.plot(np.array([low_q,m_q,high_q]).T)
plt.title('output')


plt.subplot(2, 2, 3)
low_q,m_q,high_q=get_quantiles('l')

plt.plot(np.array([low_q,m_q,high_q]).T)
plt.title('output')

plt.subplot(2, 2, 4)
cov_data_c_y=get_cov('c','y')
plt.plot(cov_data_c_y)

plt.figure()
plot_agg_var('taxes')

Y_pers = Y
panel_data_pers = panel_data