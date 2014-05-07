# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 16:14:02 2014

@author: dgevans
"""

import approximate
import numpy as np

def simulate(Para,Gamma,Y,T,T0=0):
    '''
    Simulates a sequence of state path for a given Para
    '''
    Shocks={}
    y={}
    approximate.calibrate(Para)
    t = T0+1
    while t< T:
        print t
        Gamma[t],Y[t-1], Shocks[t-1],y[t-1]= update_state(Para,Gamma[t-1])
        t += 1
    return Gamma,Y,Shocks,y

def update_state(Para,Gamma):
    '''
    Updates the state given current state gamma
    '''    
    eps = approximate.eps
    Izy = approximate.Izy
    sigma = Para.sigma_e
    
    approx = approximate.approximate(Gamma)
    dy,d2y = approx.dy,approx.d2y
    Gamma_new = np.empty(Gamma.shape)
    y=np.empty((Gamma.shape[0],Para.ny))
    epsilon = np.random.randn(len(Gamma)) * sigma        
    for z_i,z_i_new,e,y_i in zip(Gamma,Gamma_new,epsilon,y):
        #z_i_new[:] = z_i + Izy.dot(dy[eps](z_i).flatten()*e + 0.5*(d2y[eps,eps](z_i).flatten()*e**2 + d2y[sigma](z_i).flatten()*sigma**2))
        y_i[:] = approx.ss.get_y(z_i).flatten() + dy[eps](z_i).flatten()*e + 0.5*(d2y[eps,eps](z_i).flatten()*e**2 + d2y[sigma](z_i).flatten()*sigma**2).flatten()
        z_i_new[:]=Izy.dot(y_i)
        
    d2Y = approx.d2Y
    Y = approx.ss.get_Y() + 0.5*d2Y[sigma].flatten()*sigma**2
    return Para.nomalize(Gamma_new),Y,epsilon,y
    
def testEuler(Para,Gamma):
    '''
    Tests the euler equation
    '''
    eps = approximate.eps
    Izy = approximate.Izy
    sigma = Para.sigma_e
    
    approx = approximate.approximate(Gamma)
    