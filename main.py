# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 19:49:53 2014

@author: dgevans
"""
import simulate
import calibrate_begs_iid as Para
import numpy as np

Gamma,Y = {},{}

Gamma[0] = np.zeros((1000,2)) #initialize 100 agents at m = 1 for testing purposes
Gamma[0][:,0] = np.ones(1000)
Para.sigma_e = 0.3
simulate.simulate(Para,Gamma,Y,500)


