import simulate
import calibrate_begs_id_nu_ces as Para
import numpy as np
import cPickle as pickle

# INITIAL DISTRIBUTION FOR EXERCISES

print 'begin initiliazation'


N=7500
T_init=125
T_long=100
T_long_drift=75
T_imp=5
T_ss=10

Gamma,Y,Shocks,y = {},{},{},{}
Gamma[0] = np.zeros((N,3)) 
Gamma[0][:,0] = np.zeros(N)

v = simulate.v
v.execute('import numpy as np')
v.execute('import calibrate_begs_id_nu_ces as Para')
v.execute('import approximate_begs as approximate')

v.execute('old_shock_status=approximate.shock ')
v.execute('old_sigma_E=Para.sigma_E')

v.execute('approximate.shock = 0.')
v.execute('Para.sigma_E= 0.')

v.execute('approximate.calibrate(Para)')
simulate.simulate(Para,Gamma,Y,Shocks,y,T_init) #simulate 150 period with no aggregate shocks
data_initialization = Gamma,Y,Shocks,y
Gamma0 = Gamma[T_init-1]
with open('data_initialization.pickle', 'wb') as f:
    pickle.dump(data_initialization , f)
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
data_long_sample_no_agg_shock = Gamma,Y,Shocks,y
v.execute('Para.sigma_E=old_sigma_E')
v.execute('approximate.shock = old_shock_status')
v.execute('approximate.calibrate(Para)')
with open('data_long_sample_no_agg_shock.pickle', 'wb') as f:
    pickle.dump(data_long_sample_no_agg_shock , f)

print '...done long simulation without aggregate shocks'

# with aggregate shocks

print 'begin long simulation with aggregate shocks'

v.execute('np.random.set_state(state)') #use the random state
Gamma,Y,Shocks,y = {},{},{},{}
Gamma[0]=Gamma0
simulate.simulate(Para,Gamma,Y,Shocks,y,T_long)
data_long_sample_with_agg_shock = Gamma,Y,Shocks,y
with open('data_long_sample_with_agg_shock.pickle', 'wb') as f:
    pickle.dump(data_long_sample_with_agg_shock, f)


print '..done long simulation with aggregate shocks'



# EX2 Drfiting of taxes

print 'begin long drift with idiosyncratic shocks'


#Simulate all high shocks 
v.execute('approximate.shock = 1.')
v.execute('np.random.set_state(state)')
Gamma,Y,Shocks,y = {},{},{},{}
Gamma[0] = Gamma0
simulate.simulate(Para,Gamma,Y,Shocks,y,T_long_drift)
data_long_drift_high_shocks= Gamma,Y,Shocks,y
with open('data_long_drift_high_shocks.pickle', 'wb') as f:
    pickle.dump(data_long_drift_high_shocks, f)


#simulate all no shocks
v.execute('approximate.shock = 0.')
v.execute('np.random.set_state(state)')
Gamma,Y,Shocks,y = {},{},{},{}
Gamma[0] = Gamma0
simulate.simulate(Para,Gamma,Y,Shocks,y,T_long_drift)
data_long_drift_no_shocks= Gamma,Y,Shocks,y
with open('data_long_drift_no_shocks.pickle', 'wb') as f:
    pickle.dump(data_long_drift_no_shocks, f)


#simulate all low shocks
v.execute('approximate.shock = -1.')
v.execute('np.random.set_state(state)')
Gamma,Y,Shocks,y = {},{},{},{}
Gamma[0] = Gamma0
simulate.simulate(Para,Gamma,Y,Shocks,y,T_long_drift)
data_long_drift_low_shocks= Gamma,Y,Shocks,y
with open('data_long_drift_low_shocks.pickle', 'wb') as f:
    pickle.dump(data_long_drift_low_shocks, f)




print '... done long drift with idiosyncratic shocks'






# EX3 Impulse REsponse with idiosyncratic shocks
print 'begin irf with idiosyncratic shocks'

# High TFP
v.execute('np.random.set_state(state)')
Gamma[0] = Gamma0
agg_shocks=np.hstack((np.ones(T_imp),0*np.ones(T_ss)))
Gamma,Y,Shocks,y = {},{},{},{}
Gamma,Y,Shocks,y=simulate.simulate_specific_shocks_sequence(Para,Gamma0,agg_shocks)
data_irf_high_tfp_with_idiosyncratic_shocks= Gamma,Y,Shocks,y
with open('data_irf_high_tfp_with_idiosyncratic_shocks.pickle', 'wb') as f:
    pickle.dump(data_irf_high_tfp_with_idiosyncratic_shocks, f)



# Low TFP
v.execute('np.random.set_state(state)')
Gamma[0] = Gamma0
agg_shocks=np.hstack((-1*np.ones(T_imp),0*np.ones(T_ss)))
Gamma,Y,Shocks,y = {},{},{},{}
Gamma,Y,Shocks,y=simulate.simulate_specific_shocks_sequence(Para,Gamma0,agg_shocks)
data_irf_low_tfp_with_idiosyncratic_shocks= Gamma,Y,Shocks,y
with open('data_irf_low_tfp_with_idiosyncratic_shocks.pickle', 'wb') as f:
    pickle.dump(data_irf_low_tfp_with_idiosyncratic_shocks, f)


print '... done irf with idiosyncratic shocks'


# EX3 Impulse REsponse without idiosyncratic shocks

print 'begin irf drift without idiosyncratic shocks'

v.execute('Para.sigma_e[:] = 0.')
v.execute('Para.phat[:] = 0.')
# High TFP
Gamma[0] = Gamma0
agg_shocks=np.hstack((np.ones(T_imp),0*np.ones(T_ss)))
Gamma,Y,Shocks,y = {},{},{},{}
Gamma,Y,Shocks,y=simulate.simulate_specific_shocks_sequence(Para,Gamma0,agg_shocks)
data_irf_high_tfp_no_idiosyncratic_shocks= Gamma,Y,Shocks,y
with open('data_irf_high_tfp_no_idiosyncratic_shocks.pickle', 'wb') as f:
    pickle.dump(data_irf_high_tfp_no_idiosyncratic_shocks, f)



# Low TFP
Gamma[0] = Gamma0
agg_shocks=np.hstack((-1*np.ones(T_imp),0*np.ones(T_ss)))
Gamma,Y,Shocks,y = {},{},{},{}
Gamma,Y,Shocks,y=simulate.simulate_specific_shocks_sequence(Para,Gamma0,agg_shocks)
data_irf_low_tfp_no_idiosyncratic_shocks= Gamma,Y,Shocks,y
with open('data_irf_low_tfp_no_idiosyncratic_shocks.pickle', 'wb') as f:
    pickle.dump(data_irf_low_tfp_no_idiosyncratic_shocks, f)


print '...done irf without idiosyncratic shocks'


# EX4 Drfiting of taxes without idiosyncratic risk
print 'begin long drift without idiosyncratic shocks'

#Simulate all high shocks 

v.execute('approximate.shock = 1.')
v.execute('np.random.set_state(state)')
Gamma,Y,Shocks,y = {},{},{},{}
Gamma[0] = Gamma0
simulate.simulate(Para,Gamma,Y,Shocks,y,T_long_drift)
data_long_drift_high_tfp_no_idiosyncratic_shocks= Gamma,Y,Shocks,y
with open('data_long_drift_high_tfp_no_idiosyncratic_shocks.pickle', 'wb') as f:
    pickle.dump(data_long_drift_high_tfp_no_idiosyncratic_shocks, f)



#simulate all low shocks
v.execute('approximate.shock = -1.')
v.execute('np.random.set_state(state)')
Gamma,Y,Shocks,y = {},{},{},{}
Gamma[0] = Gamma0
simulate.simulate(Para,Gamma,Y,Shocks,y,T_long_drift)
data_long_drift_low_tfp_no_idiosyncratic_shocks= Gamma,Y,Shocks,y
with open('data_long_drift_low_tfp_no_idiosyncratic_shocks.pickle', 'wb') as f:
    pickle.dump(data_long_drift_low_tfp_no_idiosyncratic_shocks, f)

print '... done long drift without idiosyncratic shocks'
