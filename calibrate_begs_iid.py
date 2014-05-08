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
indx_y={'logm':0,'muhat':1,'e':2,'c' :3,'l':4,'rho1_':5,'rho2':6,'phi':7,'x_':8,'kappa_':9,'shocks':10}
indx_Y={'alpha1':0,'alpha2':1,'taxes':2,'eta':3,'lambda':4}
indx_Gamma={'m_':0,'mu_':1,'e_':2}



n = 2 # number of measurability constraints
ny = 9 # number of individual controls (m_{t},mu_{t},c_{t},l_{t},rho1_,rho2,phi,x_{t-1},kappa_{t-1}) Note that the forward looking terms are at the end
ne = 4 # number of Expectation Terms (E_t u_{c,t+1}, E_t u_{c,t+1}mu_{t+1} E_{t}x_{t-1 E_t rho_{1,t-1}} [This makes the control x_{t-1},rho_{t-1} indeed time t-1 measuable])
nY = 5 # Number of aggregates (alpha_1,alpha_2,tau,eta,lambda)
nz = 2 # Number of individual states (m_{t-1},mu_{t-1})
nv = 2 # number of forward looking terms (x_t,rho1_t)

def F(w):
    '''
    Individual first order conditions
    '''
    logm,muhat,c,l,rho1_,rho2,phi,x_,kappa_ = w[:9] #y
    EUc,EUc_mu,Ex_,Erho1_ = w[9:13] #e
    alpha1,alpha2,tau,eta,lamb = w[13:18] #Y
    logm_,muhat_ = w[18:20] #z
    x,kappa = w[20:22] #v
    eps = w[22] #shock

    m_,m = np.exp(logm_),np.exp(logm)
    
    mu_ = muhat_ * m_
    mu = muhat * m    
    
    Uc = c**(-sigma)
    Ucc = -sigma*c**(-sigma-1)
    Ul = -l**(gamma)
    Ull = -gamma*l**(gamma-1)
    
    ret = np.empty(11,dtype=w.dtype)
    ret[0] = x_ - Ex_ # x is independent of eps
    ret[1] = rho1_ - Erho1_
    
    ret[2] = x_*Uc/(beta*EUc) - Uc*c - Ul*l - x #impl
    ret[3] = alpha2 - m*Uc #defines m
    ret[4] = (1-tau)*np.exp(eps)*Uc + Ul #wage
    ret[5] = Ul - mu*(Ull*l + Ul) - phi*Ull + lamb*np.exp(eps)
    ret[6] = rho2 + kappa/Uc + eta/Uc
    ret[7] = Uc + x_*Ucc/(beta*EUc)*(mu-mu_) - mu*(Ucc*c + Uc) + rho1_*m_*Ucc/beta \
             +rho2*m*Ucc - phi*np.exp(eps)*(1-tau)*Ucc - lamb
    ret[8] = kappa_ - rho1_*EUc
    
    ret[9] = mu_*EUc - EUc_mu
    ret[10] = alpha1 - m_*EUc # bond pricing
    
    return ret
    
def G(w):
    '''
    Aggregate equations
    '''
    logm,muhat,c,l,rho1_,rho2,phi,x_,kappa_ = w[:9] #y
    eps = w[22]
 
    m = np.exp(logm)
    
    Uc = c**(-sigma)
    
    ret = np.empty(5,dtype=w.dtype)
    ret[0] = 1 - m #normalizing for Em=1
    ret[1] = c - np.exp(eps) * l # resources
    ret[2] = phi*Uc*np.exp(eps)
    ret[3] = rho1_
    ret[4] = rho2    
    
    return ret
    
def f(y):
    '''
    Expectational equations that define e=Ef(y)
    '''
    logm,muhat,c,l,rho1_,rho2,phi,x_,kappa_ = y
    
    m = np.exp(logm)
    
    mu = muhat * m
    
    Uc = c**(-sigma)    
    
    ret = np.empty(4,dtype=y.dtype)
    ret[0] = Uc
    ret[1] = Uc * mu
    ret[2] = x_
    ret[3] = rho1_
    
    return ret
    
def Finv(YSS,z):
    '''
    Given steady state YSS solves for y_i
    '''
    logm,muhat = z
    alpha1,alpha2,tau,eta,lamb = YSS
    
    m = np.exp(logm)
    
    mu = muhat * m
    
    Uc =alpha1/m
    c = (Uc)**(-1/sigma)
    Ucc = -sigma*c**(-sigma-1)
    
    Ul = -(1-tau)*Uc
    l = ( -Ul )**(1./gamma)
    Ull = -gamma*l**(gamma-1)
    
    phi = ( Ul - mu*(Ull*l + Ul) + lamb )/Ull
    
    rho1 = (Uc - mu*(Ucc*c + Uc)- phi*(1-tau)*Ucc - lamb)/ (m*Ucc*(1-1./beta))
    rho2 = -rho1
    
    x = beta*( Uc*c + Ul*l )/(1-beta)
    
    kappa = rho1*Uc
    
    return np.vstack((
    logm,muhat,c,l,rho1,rho2,phi,x,kappa   
    ))
    
def GSS(YSS,y_i):
    '''
    Aggregate conditions for the steady state
    '''
    logm,muhat,c,l,rho1,rho2,phi,x,kappa = y_i
    alpha1,alpha2,tau,eta,lamb = YSS
  
    
    Uc = c**(-sigma)
    
    return np.hstack((
    alpha1-alpha2, np.mean(c-l),eta,np.mean(phi*Uc),np.mean(rho1)    
    ))
    
def nomalize(Gamma):
    '''
    Normalizes the distriubtion of states if need be
    '''
    #in our case we want distribution of market weights to be one
    Gamma = Gamma.copy()
    Gamma[:,0] -= np.mean(Gamma[:,0])
    Gamma[:,1] -= np.mean(Gamma[:,1])
    return Gamma
    
    
    