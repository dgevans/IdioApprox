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


n = 1
ny = 4
ne = 2
nY = 2
nz = 1
nv = 1

def F(w):
    '''
    Idiosyncratic equations
    '''
    logm,c,l,x_ = w[:4]
    EUc,Ex_ = w[4:6]
    alpha1,alpha2 = w[6:8]
    logm_ = w[8]
    x = w[9]
    eps = w[10]
    
    Uc = c**(-sigma)
    Ul = -l**(gamma)
    m = np.exp(logm)    
    m_ = np.exp(logm_)
    
    
    ret = np.empty(5,dtype=w.dtype)
    ret[0] = x_ - Ex_
    ret[1] = x_*Uc/(beta*EUc) - Uc*(c-T) - Ul*l - x
    ret[2] = alpha2 - m*Uc
    ret[3] = (1-tau)*np.exp(eps)*Uc + Ul
    ret[4] = alpha1 - m_*EUc
    
    return ret
    
def G(w):
    '''
    Aggregate equations
    '''
    logm,c,l,x_ = w[:4]
    eps = w[10]
    
    ret = np.empty(2,dtype=w.dtype)
    ret[0] = 1 - np.exp(logm)
    ret[1] = c - np.exp(eps) * l
    
    return ret
    
def f(y):
    '''
    Expectational equations
    '''
    logm,c,l,x_ = y
    
    ret = np.empty(2,dtype=y.dtype)
    ret[0] = c**(-sigma)
    ret[1] = x_
    
    return ret
    
def Finv(YSS,z_i):
    '''
    Given steady state YSS solves for y_i
    '''
    logm_i = z_i
    alpha1,alpha2 = YSS
    m_i = np.exp(logm_i)    
    
    Uc_i =alpha1/m_i
    c_i = (Uc_i)**(-1/sigma)
    Ul_i = -(1-tau)*Uc_i
    l_i = ( -Ul_i )**(1./gamma)
    x_i = beta*( Uc_i*(c_i-T) + Ul_i*l_i )/(1-beta)
    
    return np.vstack((
    logm_i,c_i,l_i,x_i    
    ))
    
def GSS(YSS,y_i):
    '''
    Aggregate conditions for the steady state
    '''
    logm_i,c_i,l_i,x_i = y_i
    alpha1,alpha2 = YSS
    
    return np.hstack((
    alpha1-alpha2, np.mean(c_i-l_i)    
    ))
    
def nomalize(Gamma):
    '''
    Normalizes the distriubtion of states if need be
    '''
    #in our case we want distribution of market weights to be one
    
    return Gamma -np.log(np.mean(np.exp(Gamma)))
    
def Ftest(y,ybar,YSS,logm_,eps):
    '''
    '''
    logm,c,l,x_ = y
    alpha1,alpha2 = YSS
    logmbar,cbar,lbar,x_bar = ybar
    
    Uc = c**(-sigma)
    Ucbar = cbar**(-sigma)
    Ul = -l**(gamma)
    m = np.exp(logm) 
    x = Finv(YSS,logm)[3]
    
    
    ret = np.empty(4)
    ret[0] = x_ - x_bar
    ret[1] = x_*Uc/(beta*Ucbar) - Uc*(c-T) - Ul*l - x
    ret[2] = alpha2 - m*Uc
    ret[3] = (1-tau)*np.exp(eps)*Uc + Ul
    return ret