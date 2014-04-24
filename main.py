# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 19:49:53 2014

@author: dgevans
"""
import simulate
import calibrate_bewley_log as Para
import numpy as np

Gamma,Y = {},{}

Gamma[0] = np.zeros((100,1)) #initialize 100 agents at m = 1 for testing purposes

simulate.simulate(Para,Gamma,Y,2000)


