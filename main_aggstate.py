# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 19:49:53 2014

@author: dgevans
"""
import steadystate
import calibrate_idioinvest_ramsey as Para
from scipy.optimize import root
import numpy as np
import cPickle
from IPython.parallel import Reference

N = 10000

steadystate.calibrate(Para)
        


Gamma,Z,Y,Shocks,y = {},{},{},{},{}
Gamma[0] = np.zeros((N,3))

ss = steadystate.steadystate(zip(Gamma[0],np.ones(N)))
Z[0] = ss.get_Y()[0]

import simulate
v = simulate.v
v.execute('import calibrate_idioinvest_ramsey as Para')
v.execute('import approximate_aggstate as approximate')
v.execute('import numpy as np')
v.execute('approximate.calibrate(Para)')
simulate.simulate_aggstate(Para,Gamma,Z,Y,Shocks,y,1000) #simulate 150 period with no aggregate shocks

i = {'logc':3,'logk':5,'r':7,'pi':8}
v['i'] = i
i = Reference('i')

data = {}
data['logc'] = np.hstack(v.map(lambda y_t: np.std(y_t[:,i['logc']]),y.values()))
data['r'] =  np.hstack(v.map(lambda y_t: np.std(y_t[:,i['r']]),y.values()))
data['pi/k'] = np.hstack(v.map(lambda y_t: np.std(y_t[:,i['pi']]/np.exp(y_t[:,i['logk']])),y.values()))

simulate_data = (data,np.vstack(Y.values()),np.hstack((Z.values())))

fout = file('simulate_data.dat','w')
cPickle.dump(simulate_data,fout)
fout.close()
