# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 19:49:53 2014

@author: dgevans
"""
import steadystate
import calibrate_idioinvest_ramsey as Para
from scipy.optimize import root
import numpy as np

N = 20000

steadystate.calibrate(Para)
        


Gamma,Z,Y,Shocks,y = {},{},{},{},{}
Gamma[0] = np.zeros((N,3))

ss = steadystate.steadystate(Gamma[0].T)
Z[0] = ss.get_Y()[0]

import simulate
v = simulate.v
v.execute('import calibrate_idioinvest_ramsey as Para')
v.execute('import approximate_aggstate as approximate')
v.execute('approximate.calibrate(Para)')
simulate.simulate_aggstate(Para,Gamma,Z,Y,Shocks,y,2000) #simulate 150 period with no aggregate shocks