# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 12:18:27 2014

@author: dgevans
"""
import numpy as np

beta = 0.95
sigma = 2.
gamma = 2.
sigma_e = 0.1
T = 0.2
tau = 0.2


n = 1 # number of measurability constraints
ny = 4 # number of individual controls (m_{t},c_{t},l_{t},x_{t-1}) Note that the forward looking terms are at the end
ne = 2 # number of Expectation Terms (E_t u_{c,t+1} and E_{t}x_{t-1} [This makes the control x_{t-1} indeed time t-1 measuable])
nY = 2 # Number of aggregates (alpha_1,alpha_2)
nz = 1 # Number of individual states (m_{t-1})
nv = 1 # number of forward looking terms (x_t)

def F(w):
    '''
    Individual first order conditions
    '''
    m,c,l,x_ = w[:4] #y
    EUc,Ex_ = w[4:6] #e
    alpha1,alpha2 = w[6:8] #Y
    m_ = w[8] #z
    x = w[9] #v
    eps = w[10] #shock
    
    Uc = c**(-sigma)
    Ul = -l**(gamma)
    
    ret = np.empty(5,dtype=w.dtype)
    ret[0] = x_ - Ex_ # x is independent of eps
    ret[1] = x_*Uc/(beta*EUc) - Uc*(c-T) - Ul*l - x #impl
    ret[2] = alpha2 - m*Uc #defines m
    ret[3] = (1-tau)*np.exp(eps)*Uc + Ul #wage
    ret[4] = alpha1 - m_*EUc # bond pricing
    
    return ret
    
def G(w):
    '''
    Aggregate equations
    '''
    m,c,l,x_ = w[:4]
    eps = w[10]
    
    ret = np.empty(2,dtype=w.dtype)
    ret[0] = 1 - m #normalizing for Em=1
    ret[1] = c - np.exp(eps) * l # resources
    
    return ret
    
def f(y):
    '''
    Expectational equations that define e=Ef(y)
    '''
    m,c,l,x_ = y
    
    ret = np.empty(2,dtype=y.dtype)
    ret[0] = c**(-sigma)
    ret[1] = x_
    
    return ret
    
def Finv(YSS,z_i):
    '''
    Given steady state YSS solves for y_i
    '''
    m_i = z_i
    alpha1,alpha2 = YSS
    Uc_i =alpha1/m_i#np.exp(logm_i)
    c_i = (Uc_i)**(-1/sigma)
    Ul_i = -(1-tau)*Uc_i
    l_i = ( -Ul_i )**(1./gamma)
    x_i = beta*( Uc_i*(c_i-T) + Ul_i*l_i )/(1-beta)
    
    return np.vstack((
    m_i,c_i,l_i,x_i    
    ))
    
def GSS(YSS,y_i):
    '''
    Aggregate conditions for the steady state
    '''
    m_i,c_i,l_i,x_i = y_i
    alpha1,alpha2 = YSS
    
    return np.hstack((
    alpha1-alpha2, np.mean(c_i-l_i)    
    ))
    
def nomalize(Gamma):
    '''
    Normalizes the distriubtion of states if need be
    '''
    #in our case we want distribution of market weights to be one
    
    return Gamma/np.mean(Gamma)
    
    
    