# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 19:49:53 2014

@author: dgevans
"""
import simulate
import calibrate_begs_iid as Para
import numpy as np


Gamma,Y,Shocks,y = {},{},{},{}

Gamma[0] = np.zeros((100,2)) #initialize 100 agents at m = 1 for testing purposes
Gamma[0][:,0] = np.zeros(100)


v = simulate.v
v.execute('import calibrate_begs_iid as Para')
v.execute('import approximate_parallel as approximate')
v.execute('approximate.calibrate(Para)')
simulate.simulate(Para,Gamma,Y,Shocks,y,10)
