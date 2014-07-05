import simulate
import calibrate_begs_id_nu_ghh as Para
import numpy as np
import cPickle as pickle
import pandas as pd
import os as os


N=30000
T_init=10
T_long=100
T_long_drift=30
T_imp=5
T_ss=10
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
        simulation_data[t]=np.hstack((output[t],np.atleast_2d(mu[t]).reshape(N,1),y[t]))
        simulation_data[t]=np.hstack((output[t],np.atleast_2d(mu[t]).reshape(N,1),y[t]))
        panel_data=pd.Panel(simulation_data,minor_axis=['y','mu','logm','muhat','e','c' ,'l','rho1_','rho2','phi','wages','UcP','a','I','x_','kappa_'])
    return panel_data

    
def get_quantiles(var,panel_data):
        var_data=panel_data.minor_xs(var)
        low_q=var_data.quantile(.10,axis=0)
        m_q=var_data.quantile(.5,axis=0)
        high_q=var_data.quantile(.9,axis=0)            
        return np.array(low_q),np.array(m_q),np.array(high_q)
    
    
def get_cov(var_1,var_2,panel_data):
        T=np.shape(panel_data)[0]            

        cov_data= np.array(map(lambda t: np.corrcoef(panel_data.minor_xs(var_1)[t][panel_data.minor_xs(var_1)[t]<panel_data.minor_xs(var_1)[t].quantile(truncation)],panel_data.minor_xs(var_2)[t][panel_data.minor_xs(var_2)[t]<panel_data.minor_xs(var_2)[t].quantile(truncation)])[0,1],range(T-1)))
        
        return cov_data
    

    
def get_var(variable,panel_data):
        T=np.shape(panel_data)[0]

        var_data= np.array(map(lambda t:  np.std(panel_data.minor_xs(variable)[t][panel_data.minor_xs(variable)[t]<panel_data.minor_xs(variable)[t].quantile(truncation)])/(np.mean(panel_data.minor_xs(variable)[t][panel_data.minor_xs(variable)[t]<panel_data.minor_xs(variable)[t].quantile(truncation)])),range(T-1)))
        
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
    debt=np.array(panel_data.minor_xs('a').sum())/len(panel_data.major_axis)

    
    store_data=Y,low_q_c,m_q_c,high_q_c,low_q_y,m_q_y,high_q_y,low_q_l,m_q_l,high_q_l,low_q_a,m_q_a,high_q_a,cov_data_c_y,cov_data_a_y,cov_data_a_c,cov_data_l_c,var_data_c,var_data_a,var_data_l,var_data_y,debt
    
    with open(nameoffile, 'wb') as f:
        pickle.dump(store_data, f)
    
    


# INITIAL DISTRIBUTION FOR EXERCISES

print 'begin initiliazation'
import steadystate
from scipy.optimize import root
steadystate.calibrate(Para)
steadystate.Finv = Para.time0Finv
steadystate.GSS = Para.time0GSS
 
#match  moments
 
def f(z):
    mu = z[:2]
    Sigma = np.array([[z[2],z[3]],[z[3],z[4]]])
    Sig = np.diag(Sigma)
    mu_y,mu_a = np.exp(mu + 0.5*Sig)
    Sigma2 = np.exp(mu+mu.reshape(-1,1) + 0.5*(Sig+Sig.reshape(-1,1)))*(np.exp(Sigma)-1)
    sig_y = np.sqrt(Sigma2[0,0])
    sig_a = np.sqrt(Sigma2[1,1])
    corr_ya = Sigma2[0,1]/(sig_y*sig_a)
    return np.array([mu_y,mu_a,sig_y/mu_y,sig_a/mu_a,corr_ya]) - np.array([1.,5.,2.65,6.53,0.5])
     
z = root(f,np.array([1.,1,1.,0.,1.])).x
mu = z[:2]
Sigma = np.array([[z[2],z[3]],[z[3],z[4]]])
Gamma_ya = np.exp(np.random.multivariate_normal(mu,Sigma,N))
beta,sigma,gamma,psi = Para.beta,Para.sigma,Para.gamma,Para.psi
T,tau = 0.2,0.3
y,a = Gamma_ya.T
c = (1-beta)*a/beta + T + (1-tau)*y

l = (psi*(1-tau)*y)**(1/(1+gamma))
e = np.log(y/l)
chat = c-psi*l**(1+gamma)/(1+gamma)
Uc = chat**(-sigma)

logm = -np.log(Uc)
logm -= logm.mean()
Gamma0= np.vstack([logm,e]).T


        
ss = steadystate.steadystate(Gamma0.T)

Gamma,Y,Shocks,y = {},{},{},{}
Gamma[0] = ss.get_y(ss.z_i)[:3,:].T
 


v = simulate.v
v.execute('import numpy as np')
v.execute('import calibrate_begs_id_nu_ghh as Para')
v.execute('import approximate_begs as approximate')

v.execute('old_shock_status=approximate.shock ')
v.execute('old_sigma_E=Para.sigma_E')

v.execute('approximate.shock = 0.')
v.execute('Para.sigma_E= 0.')

v.execute('approximate.calibrate(Para)')
simulate.simulate(Para,Gamma,Y,Shocks,y,T_init) #simulate 150 period with no aggregate shocks
data_initialization = Y,y
Gamma0 = Gamma[T_init-1]
with open('data_initialization.pickle', 'wb') as f:
    pickle.dump(data_initialization , f)
save_plot_data(data_initialization,'plot_data_initialization.pickle')
save_plot_data(data_initialization,'plot_data_initialization.pickle')
v.execute('Para.sigma_E=old_sigma_E')
v.execute('approximate.shock = old_shock_status')
v.execute('approximate.calibrate(Para)')
    
    

print '....initialized'




# EX1 Long run simulations with and without aggregate shocks


print 'begin long simulation without aggregate shocks'

v.execute('state = np.random.get_state()') #save random state
# without aggregate shocks
v.execute('old_shock_status=approximate.shock ')
v.execute('old_sigma_E=Para.sigma_E')
v.execute('approximate.shock = 0.')
v.execute('Para.sigma_E= 0.')
Gamma,Y,Shocks,y = {},{},{},{}
Gamma[0]=Gamma0
simulate.simulate(Para,Gamma,Y,Shocks,y,T_long) #Keep off Aggregate shocks
data_long_sample_no_agg_shock = Y,y
v.execute('Para.sigma_E=old_sigma_E')
v.execute('approximate.shock = old_shock_status')
v.execute('approximate.calibrate(Para)')
with open('data_long_sample_no_agg_shock.pickle', 'wb') as f:
    pickle.dump(data_long_sample_no_agg_shock , f)
save_plot_data(data_long_sample_no_agg_shock,'plot_data_long_sample_no_agg_shock.pickle')
print '...done long simulation without aggregate shocks'

# with aggregate shocks

print 'begin long simulation with aggregate shocks'

v.execute('np.random.set_state(state)') #use the random state
Gamma,Y,Shocks,y = {},{},{},{}
Gamma[0]=Gamma0
simulate.simulate(Para,Gamma,Y,Shocks,y,T_long)
data_long_sample_with_agg_shock = Y,y
with open('data_long_sample_with_agg_shock.pickle', 'wb') as f:
    pickle.dump(data_long_sample_with_agg_shock, f)

save_plot_data(data_long_sample_with_agg_shock,'plot_data_long_sample_with_agg_shock.pickle')
print '..done long simulation with aggregate shocks'



# EX2 Drfiting of taxes

print 'begin long drift with idiosyncratic shocks'


#Simulate all high shocks 
v.execute('approximate.shock = 1.')
v.execute('np.random.set_state(state)')
Gamma,Y,Shocks,y = {},{},{},{}
Gamma[0] = Gamma0
simulate.simulate(Para,Gamma,Y,Shocks,y,T_long_drift)
data_long_drift_high_shocks= Y,y
with open('data_long_drift_high_shocks.pickle', 'wb') as f:
    pickle.dump(data_long_drift_high_shocks, f)
save_plot_data(data_long_drift_high_shocks,'plot_data_long_drift_high_shocks.pickle')


#==============================================================================
# 
# #simulate all no shocks
# v.execute('approximate.shock = 0.')
# v.execute('np.random.set_state(state)')
# Gamma,Y,Shocks,y = {},{},{},{}
# Gamma[0] = Gamma0
# simulate.simulate(Para,Gamma,Y,Shocks,y,T_long_drift)
# data_long_drift_no_shocks= Y,y
# with open('data_long_drift_no_shocks.pickle', 'wb') as f:
#     pickle.dump(data_long_drift_no_shocks, f)
# 
# save_plot_data(data_long_drift_no_shocks,'plot_data_long_drift_no_shocks.pickle')
# 
# 
# #simulate all low shocks
# v.execute('approximate.shock = -1.')
# v.execute('np.random.set_state(state)')
# Gamma,Y,Shocks,y = {},{},{},{}
# Gamma[0] = Gamma0
# simulate.simulate(Para,Gamma,Y,Shocks,y,T_long_drift)
# data_long_drift_low_shocks= Y,y
# with open('data_long_drift_low_shocks.pickle', 'wb') as f:
#     pickle.dump(data_long_drift_low_shocks, f)
# save_plot_data(data_long_drift_low_shocks,'plot_data_long_drift_low_shocks.pickle')
#==============================================================================



print '... done long drift with idiosyncratic shocks'






# EX3 Impulse REsponse with idiosyncratic shocks
print 'begin irf with idiosyncratic shocks'

# High TFP
v.execute('np.random.set_state(state)')
Gamma[0] = Gamma0
agg_shocks=np.hstack((np.ones(T_imp),0*np.ones(T_ss)))
Gamma,Y,Shocks,y = {},{},{},{}
Gamma,Y,Shocks,y=simulate.simulate_specific_shocks_sequence(Para,Gamma0,agg_shocks)
data_irf_high_tfp_with_idiosyncratic_shocks= Y,y
with open('data_irf_high_tfp_with_idiosyncratic_shocks.pickle', 'wb') as f:
    pickle.dump(data_irf_high_tfp_with_idiosyncratic_shocks, f)

save_plot_data(data_irf_high_tfp_with_idiosyncratic_shocks,'plot_data_irf_high_tfp_with_idiosyncratic_shocks.pickle')
#==============================================================================
# 
# 
# # Low TFP
# v.execute('np.random.set_state(state)')
# Gamma[0] = Gamma0
# agg_shocks=np.hstack((-1*np.ones(T_imp),0*np.ones(T_ss)))
# Gamma,Y,Shocks,y = {},{},{},{}
# Gamma,Y,Shocks,y=simulate.simulate_specific_shocks_sequence(Para,Gamma0,agg_shocks)
# data_irf_low_tfp_with_idiosyncratic_shocks= Y,y
# with open('data_irf_low_tfp_with_idiosyncratic_shocks.pickle', 'wb') as f:
#     pickle.dump(data_irf_low_tfp_with_idiosyncratic_shocks, f)
# 
# save_plot_data(data_irf_low_tfp_with_idiosyncratic_shocks,'plot_data_irf_low_tfp_with_idiosyncratic_shocks.pickle')
# 
# print '... done irf with idiosyncratic shocks'
#==============================================================================


# EX3 Impulse REsponse without idiosyncratic shocks

print 'begin irf drift without idiosyncratic shocks'

v.execute('Para.sigma_e[:] = 0.')
v.execute('Para.phat[:] = 0.')
# High TFP
Gamma[0] = Gamma0
agg_shocks=np.hstack((np.ones(T_imp),0*np.ones(T_ss)))
Gamma,Y,Shocks,y = {},{},{},{}
Gamma,Y,Shocks,y=simulate.simulate_specific_shocks_sequence(Para,Gamma0,agg_shocks)
data_irf_high_tfp_no_idiosyncratic_shocks= Y,y
with open('data_irf_high_tfp_no_idiosyncratic_shocks.pickle', 'wb') as f:
    pickle.dump(data_irf_high_tfp_no_idiosyncratic_shocks, f)

save_plot_data(data_irf_high_tfp_no_idiosyncratic_shocks,'plot_data_irf_high_tfp_no_idiosyncratic_shocks.pickle')

#==============================================================================
# 
# # Low TFP
# Gamma[0] = Gamma0
# agg_shocks=np.hstack((-1*np.ones(T_imp),0*np.ones(T_ss)))
# Gamma,Y,Shocks,y = {},{},{},{}
# Gamma,Y,Shocks,y=simulate.simulate_specific_shocks_sequence(Para,Gamma0,agg_shocks)
# data_irf_low_tfp_no_idiosyncratic_shocks= Y,y
# with open('data_irf_low_tfp_no_idiosyncratic_shocks.pickle', 'wb') as f:
#     pickle.dump(data_irf_low_tfp_no_idiosyncratic_shocks, f)
# 
# save_plot_data(data_irf_low_tfp_no_idiosyncratic_shocks,'plot_data_irf_low_tfp_no_idiosyncratic_shocks.pickle')
# 
# print '...done irf without idiosyncratic shocks'
#==============================================================================


# EX4 Drfiting of taxes without idiosyncratic risk
print 'begin long drift without idiosyncratic shocks'

#Simulate all high shocks 

v.execute('approximate.shock = 1.')
v.execute('np.random.set_state(state)')
Gamma,Y,Shocks,y = {},{},{},{}
Gamma[0] = Gamma0
simulate.simulate(Para,Gamma,Y,Shocks,y,T_long_drift)
data_long_drift_high_tfp_no_idiosyncratic_shocks= Y,y
with open('data_long_drift_high_tfp_no_idiosyncratic_shocks.pickle', 'wb') as f:
    pickle.dump(data_long_drift_high_tfp_no_idiosyncratic_shocks, f)

save_plot_data(data_long_drift_high_tfp_no_idiosyncratic_shocks,'plot_data_long_drift_high_tfp_no_idiosyncratic_shocks.pickle')

#==============================================================================
# 
# #simulate all low shocks
# v.execute('approximate.shock = -1.')
# v.execute('np.random.set_state(state)')
# Gamma,Y,Shocks,y = {},{},{},{}
# Gamma[0] = Gamma0
# simulate.simulate(Para,Gamma,Y,Shocks,y,T_long_drift)
# data_long_drift_low_tfp_no_idiosyncratic_shocks= Y,y
# with open('data_long_drift_low_tfp_no_idiosyncratic_shocks.pickle', 'wb') as f:
#     pickle.dump(data_long_drift_low_tfp_no_idiosyncratic_shocks, f)
# 
# print '... done long drift without idiosyncratic shocks'
# save_plot_data(data_long_drift_low_tfp_no_idiosyncratic_shocks,'plot_data_long_drift_low_tfp_no_idiosyncratic_shocks.pickle')
#==============================================================================


os.system('mv *.pickle plot_data/')
execfile('plot_simulations.py')