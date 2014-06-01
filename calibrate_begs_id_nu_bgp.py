# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 12:18:27 2014

@author: dgevans
"""
import numpy as np

beta = 0.95
gamma = 2.
psi=0.5
sigma_e = 0.04
sigma_E = 0.0


n = 2 # number of measurability constraints
nG = 2 # number of aggregate measurability constraints.
ny = 10 # number of individual controls (m_{t},mu_{t},c_{t},l_{t},rho1_,rho2,phi,x_{t-1},kappa_{t-1}) Note that the forward looking terms are at the end
ne = 4 # number of Expectation Terms (E_t u_{c,t+1}, E_t u_{c,t+1}mu_{t+1} E_{t}x_{t-1 E_t rho_{1,t-1}} [This makes the control x_{t-1},rho_{t-1} indeed time t-1 measuable])
nY = 6 # Number of aggregates (alpha_1,alpha_2,tau,eta,lambda)
nz = 3 # Number of individual states (m_{t-1},mu_{t-1})
nv = 2 # number of forward looking terms (x_t,rho1_t)
n_p = 1 #number of parameters

phat = np.array([-0.03])

indx_y={'logm':0,'muhat':1,'e':2,'c' :3,'l':4,'rho1_':5,'rho2':6,'phi':7,'x_':8,'kappa_':9,'shocks':10}
indx_Y={'alpha1':0,'alpha2':1,'taxes':2,'eta':3,'lambda':4}
indx_Gamma={'m_':0,'mu_':1,'e_':2}


def F(w):
    '''
    Individual first order conditions
    '''
    logm,muhat,e,c,l,rho1_,rho2,phi,x_,kappa_ = w[:ny] #y
    EUc,EUc_mu,Ex_,Erho1_ = w[ny:ny+ne] #e
    alpha1,alpha2,tau,eta,lamb,T = w[ny+ne:ny+ne+nY] #Y
    logm_,muhat_,e_= w[ny+ne+nY:ny+ne+nY+nz] #z
    x,kappa = w[ny+ne+nY+nz:ny+ne+nY+nz+nv] #v
    nu = w[ny+ne+nY+nz+nv+n_p-1] #v
    eps = w[ny+ne+nY+nz+nv+n_p] #shock
    Eps = w[ny+ne+nY+nz+nv+n_p+1] #aggregate shock
    
    m_,m = np.exp(logm_),np.exp(logm)

    mu_ = muhat_ * m_
    mu = muhat * m    
    
    Uc = psi/c
    Ucc = -psi/(c**2)
    Ul = -(1-psi)/(1-l)    
    Ull = -(1-psi)/((1-l)**2)    
    ret = np.empty(ny+n,dtype=w.dtype)
    ret[0] = x_ - Ex_ # x is independent of eps
    ret[1] = rho1_ - Erho1_
    
    ret[2] = x_*Uc/(beta*EUc) - Uc*(c-T) - Ul*l - x #impl
    ret[3] = alpha2 - m*Uc #defines m
    ret[4] = (1-tau)*np.exp(e)*Uc + Ul #wage
    ret[5] = Ul - mu*(Ull*l + Ul) - phi*Ull + lamb*np.exp(e)
    ret[6] = rho2 + kappa/Uc + eta/Uc
    ret[7] = Uc + x_*Ucc/(beta*EUc)*(mu-mu_) - mu*(Ucc*c + Uc) + rho1_*m_*Ucc/beta \
             +rho2*m*Ucc - phi*np.exp(e)*(1-tau)*Ucc - lamb
    ret[8] = kappa_ - rho1_*EUc
    ret[9] = e - nu*e_ - eps
    
    ret[10] = mu_*EUc - EUc_mu
    ret[11] = alpha1 - m_*EUc # bond pricing
    
    return ret
    
def G(w):
    '''
    Aggregate equations
    '''
    logm,muhat,e,c,l,rho1_,rho2,phi,x_,kappa_ = w[:ny] #y
    alpha1,alpha2,tau,eta,lamb,T = w[ny+ne:ny+ne+nY] #Y
    Eps = w[ny+ne+nY+nz+nv+n_p+1]
 
    m = np.exp(logm)
    Uc = psi/c
    
    ret = np.empty(nY+nG,dtype=w.dtype)
    ret[0] = alpha1 #alpha_1 can't depend on Eps
    ret[1] = muhat # muhat must integrate to zero for all Eps
    
    ret[2] = 1 - m #normalizing for Em=1
    ret[3] = c + 0.17*np.exp(Eps) - np.exp(e) * l # resources
    ret[4] = phi*Uc*np.exp(e+Eps)
    ret[5] = rho2    
    
    ret[6] = T # normalize average transfers to zero note doesn't depend on Eps here
    ret[7] = rho1_
    
    return ret
    
def f(y):
    '''
    Expectational equations that define e=Ef(y)
    '''
    logm,muhat,e,c,l,rho1_,rho2,phi,x_,kappa_ = y
    
    m = np.exp(logm)
    mu = muhat * m
    
    Uc = psi/c
    
    ret = np.empty(ne,dtype=y.dtype)
    ret[0] = Uc
    ret[1] = Uc * mu
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
    
    return np.vstack((
    logm,muhat,e,c,l,rho1,rho2,phi,x,kappa   
    ))
    
def GSS(YSS,y_i):
    '''
    Aggregate conditions for the steady state
    '''
    logm,muhat,e,c,l,rho1,rho2,phi,x,kappa = y_i
    alpha1,alpha2,tau,eta,lamb,T = YSS
  
    
    Uc = psi/c
    
    return np.hstack((
    alpha1-alpha2, np.mean(c-l*np.exp(e)),eta,np.mean(phi*Uc*np.exp(e)),np.mean(rho1),T    
    ))
    
def nomalize(Gamma):
    '''
    Normalizes the distriubtion of states if need be
    '''
    #in our case we want distribution of market weights to be one
    trunc = np.where(Gamma[:,0]<-3.5)[0]
    if len(trunc)>0:
        Gamma[trunc] = np.zeros((len(trunc),3))
    Gamma[:,0] -= np.mean(Gamma[:,0])
    Gamma[:,1] -= np.mean(Gamma[:,1])
    return Gamma
    
    
    