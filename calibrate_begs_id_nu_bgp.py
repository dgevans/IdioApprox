# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 12:18:27 2014

@author: dgevans
"""
import numpy as np

beta = 0.95
gamma = 2.
psi=0.6994
sigma_e = np.array([0.065,0.2])
sigma_E = 0.025


ll = np.array([-np.inf,-np.inf,-np.inf])
ul = np.array([np.inf,np.inf,np.inf])

n = 2 # number of measurability constraints
nG = 2 # number of aggregate measurability constraints.
ny = 12 # number of individual controls (m_{t},mu_{t},c_{t},l_{t},rho1_,rho2,phi,x_{t-1},kappa_{t-1}) Note that the forward looking terms are at the end
ne = 4 # number of Expectation Terms (E_t u_{c,t+1}, E_t u_{c,t+1}mu_{t+1} E_{t}x_{t-1 E_t rho_{1,t-1}} [This makes the control x_{t-1},rho_{t-1} indeed time t-1 measuable])
nY = 6 # Number of aggregates (alpha_1,alpha_2,tau,eta,lambda)
nz = 3 # Number of individual states (m_{t-1},mu_{t-1})
nv = 2 # number of forward looking terms (x_t,rho1_t)
n_p = 1 #number of parameters
neps = len(sigma_e)

phat = np.array([-0.01])

indx_y={'logm':0,'muhat':1,'e':2,'c' :3,'l':4,'rho1_':5,'rho2':6,'phi':7,'x_':8,'kappa_':9,'shocks':10}
indx_Y={'alpha1':0,'alpha2':1,'taxes':2,'eta':3,'lambda':4}
indx_Gamma={'m_':0,'mu_':1,'e_':2}


def F(w):
    '''
    Individual first order conditions
    '''
    logm,muhat,e,c,l,rho1_,rho2,phi,w_e,UcP,x_,kappa_ = w[:ny] #y
    EUcP,EUc_muP,Ex_,Erho1_ = w[ny:ny+ne] #e
    alpha1,alpha2,tau,eta,lamb,T = w[ny+ne:ny+ne+nY] #Y
    logm_,muhat_,e_= w[ny+ne+nY:ny+ne+nY+nz] #z
    x,kappa = w[ny+ne+nY+nz:ny+ne+nY+nz+nv] #v
    nu = w[ny+ne+nY+nz+nv+n_p-1] #v
    eps_p,eps_t = w[ny+ne+nY+nz+nv+n_p:ny+ne+nY+nz+nv+n_p+neps] #shock
    Eps = w[ny+ne+nY+nz+nv+n_p+neps] #aggregate shock
    
    m_,m = np.exp(logm_),np.exp(logm)

    mu_ = muhat_ * m_
    mu = muhat * m    
    
    P = 1. + 0.0*Eps #payoff shock
    
    Uc = psi/c
    Ucc = -psi/(c**2)
    Ul = -(1-psi)/(1-l)    
    Ull = -(1-psi)/((1-l)**2)    
    ret = np.empty(ny+n,dtype=w.dtype)
    ret[0] = x_ - Ex_ # x is independent of eps
    ret[1] = rho1_ - Erho1_
    
    ret[2] = x_*UcP/(beta*EUcP) - Uc*(c-T) - Ul*l - x #impl
    ret[3] = alpha2 - m*Uc #defines m
    ret[4] = (1-tau)*w_e*Uc + Ul #wage
    ret[5] = Ul - mu*(Ull*l + Ul) - phi*Ull + lamb*w_e
    ret[6] = rho2 + kappa/Uc + eta/Uc
    ret[7] = Uc + x_*Ucc*P/(beta*EUcP)*(mu-mu_) - mu*(Ucc*c + Uc) + rho1_*m_*Ucc/beta \
             +rho2*m*Ucc - phi*w_e*(1-tau)*Ucc - lamb
    ret[8] = kappa_ - rho1_*EUcP
    ret[9] = e - nu*e_ - eps_p
    ret[10] = w_e - np.exp(Eps+e+eps_t)
    ret[11] = UcP - Uc*P
    
    ret[12] = mu_*EUcP - EUc_muP
    ret[13] = alpha1 - m_*EUcP # bond pricing
    
    return ret
    
def G(w):
    '''
    Aggregate equations
    '''
    logm,muhat,e,c,l,rho1_,rho2,phi,w_e,UcP,x_,kappa_ = w[:ny] #y
    alpha1,alpha2,tau,eta,lamb,T = w[ny+ne:ny+ne+nY] #Y
    Eps = w[ny+ne+nY+nz+nv+n_p+1]
 
    m = np.exp(logm)
    Uc = psi/c
    
    ret = np.empty(nY+nG,dtype=w.dtype)
    ret[0] = alpha1 #alpha_1 can't depend on Eps
    ret[1] = muhat # muhat must integrate to zero for all Eps
    
    ret[2] = 0 - logm #normalizing for Em=1
    ret[3] = c + 0.17 - w_e * l # resources
    ret[4] = phi*Uc*w_e
    ret[5] = rho2    
    
    ret[6] = T # normalize average transfers to zero note doesn't depend on Eps here
    ret[7] = rho1_
    
    return ret
    
def f(y):
    '''
    Expectational equations that define e=Ef(y)
    '''
    logm,muhat,e,c,l,rho1_,rho2,phi,w_e,UcP,x_,kappa_ = y
    
    m = np.exp(logm)
    mu = muhat * m
    
    Uc = psi/c
    
    ret = np.empty(ne,dtype=y.dtype)
    ret[0] = UcP
    ret[1] = UcP * mu
    ret[2] = x_
    ret[3] = rho1_
    
    return ret
    
def Finv(YSS,z):
    '''
    Given steady state YSS solves for y_i
    '''
    logm,muhat,e = z
    alpha1,alpha2,tau,eta,lamb,T = YSS
    
    m = np.exp(logm)
    mu = muhat * m
    
    Uc =alpha1/m
    c = psi/Uc
    Ucc =-psi/(c**2)    

    Ul = -(1-tau)*Uc*np.exp(e)
    l = 1+(1-psi)/Ul
    Ull = -(1-psi)/((1-l)**2)
    
    phi = ( Ul - mu*(Ull*l + Ul) + lamb*np.exp(e) )/Ull
    
    rho1 = (Uc - mu*(Ucc*c + Uc)- phi*np.exp(e)*(1-tau)*Ucc - lamb)/ (m*Ucc*(1-1./beta))
    rho2 = -rho1
    
    x = beta*( Uc*c + Ul*l )/(1-beta)
    
    kappa = rho1*Uc
    
    w_e = np.exp(e)    
    
    return np.vstack((
    logm,muhat,e,c,l,rho1,rho2,phi,w_e,Uc,x,kappa   
    ))
    
def GSS(YSS,y_i):
    '''
    Aggregate conditions for the steady state
    '''
    logm,muhat,e,c,l,rho1,rho2,phi,w_e,UcP,x,kappa = y_i
    alpha1,alpha2,tau,eta,lamb,T = YSS
  
    
    Uc = psi/c
    
    return np.hstack((
    alpha1-alpha2, np.mean(c-l*w_e),eta,np.mean(phi*Uc*w_e),np.mean(rho1),T    
    ))
    
def nomalize(Gamma):
    '''
    Normalizes the distriubtion of states if need be
    '''
    #in our case we want distribution of market weights to be one
    for i in range(nz):
        if(len(np.where(Gamma[:,i]<ll[i])[0])):
            Gamma[np.where(Gamma[:,i]<ll[i])[0],i] = ll[i]
        if(len(np.where(Gamma[:,i]>ul[i])[0])):
            Gamma[np.where(Gamma[:,i]>ul[i])[0],i] = ul[i]
                
    Gamma[:,0] -= np.mean(Gamma[:,0])
    Gamma[:,1] -= np.mean(Gamma[:,1])#/np.exp(Gamma[:,0]))*np.exp(Gamma[:,0])
    return Gamma
    
    
    