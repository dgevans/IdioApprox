# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 12:18:27 2014

@author: dgevans
"""
import numpy as np
import pycppad as ad

beta = 0.95
gamma = 1.
sigma = 0.5
sigma_e = np.array([0.12,0.2])
sigma_E = 0.03
chi = 0.
mu_e = 0.7
psi = 10.

ll = np.array([-np.inf,-np.inf,-np.inf])
ul = np.array([np.inf,np.inf,np.inf])

n = 2 # number of measurability constraints
nG = 2 # number of aggregate measurability constraints.
ny = 14 # number of individual controls (m_{t},mu_{t},c_{t},l_{t},rho1_,rho2,phi,x_{t-1},kappa_{t-1}) Note that the forward looking terms are at the end
ne = 4 # number of Expectation Terms (E_t u_{c,t+1}, E_t u_{c,t+1}mu_{t+1} E_{t}x_{t-1 E_t rho_{1,t-1}} [This makes the control x_{t-1},rho_{t-1} indeed time t-1 measuable])
nY = 6 # Number of aggregates (alpha_1,alpha_2,tau,eta,lambda)
nz = 3 # Number of individual states (m_{t-1},mu_{t-1})
nv = 2 # number of forward looking terms (x_t,rho1_t)
n_p = 1 #number of parameters
neps = len(sigma_e)

phat = np.array([-0.01])

indx_y={'logm':0,'muhat':1,'e':2,'c' :3,'l':4,'rho1_':5,'rho2':6,'phi':7,'wages':8,'UcP':9,'a':10,'x_':11,'kappa_':12,'pers_shock':13,'trans_shock':14}
indx_Y={'alpha1':0,'alpha2':1,'taxes':2,'eta':3,'lambda':4,'T':5,'shock':6}
indx_Gamma={'m_':0,'mu_':1,'e_':2}


def F(w):
    '''
    Individual first order conditions
    '''
    logm,muhat,e,c,l,rho1_,rho2,phi,w_e,UcP,a,I,x_,kappa_ = w[:ny] #y
    EUcP,EUc_muP,Ex_,Erho1_ = w[ny:ny+ne] #e
    alpha1,alpha2,tau,eta,lamb,T = w[ny+ne:ny+ne+nY] #Y
    logm_,muhat_,e_= w[ny+ne+nY:ny+ne+nY+nz] #z
    x,kappa = w[ny+ne+nY+nz:ny+ne+nY+nz+nv] #v
    nu = w[ny+ne+nY+nz+nv+n_p-1] #v
    eps_p,eps_t = w[ny+ne+nY+nz+nv+n_p:ny+ne+nY+nz+nv+n_p+neps] #shock
    Eps = w[ny+ne+nY+nz+nv+n_p+neps] #aggregate shock
    
    if (ad.value(logm) < -3.5):
        shock = 0.
    else:
        shock = 1.
    
    m_,m = np.exp(logm_),np.exp(logm)

    mu_ = muhat_ * m_
    mu = muhat * m    
    
    P = 1. + chi*Eps #payoff shock
    
    Uc = c**(-sigma)
    Ucc = -sigma*c**(-sigma-1)
    Ul = -psi*l**(gamma)
    Ull = -psi*gamma*l**(gamma-1)
    
    ret = np.empty(ny+n,dtype=w.dtype)
    ret[0] = x_ - Ex_ # x is independent of eps
    ret[1] = rho1_ - Erho1_
    
    ret[2] = x_*UcP/(beta*EUcP) - Uc*(c-T) - Ul*l - x #impl
    ret[3] = alpha2 - m*Uc #defines m
    ret[4] = (1-tau)*w_e*Uc + Ul #wage
    ret[5] = Ul - mu*(Ull*l + Ul) - phi*Ull + lamb*w_e
    ret[6] = rho2 + kappa/Uc + eta/Uc
    ret[7] = Uc + x_*Ucc*P/(beta*EUcP)*(mu-mu_) - mu*(Ucc*(c-T) + Uc) + rho1_*m_*Ucc/beta \
             +rho2*m*Ucc - phi*w_e*(1-tau)*Ucc - lamb
    ret[8] = kappa_ - rho1_*EUcP
    ret[9] = e - ( shock*(nu*e_+eps_p + (1-nu)*mu_e) + (1-shock)*e_  )
    ret[10] = w_e - np.exp(e+shock*eps_t+Eps)
    ret[11] = UcP - Uc*P
    ret[12] = a - x_/(beta*EUcP)
    ret[13] = I - x_*P/(beta*EUcP) - T
    
    
    ret[14] = mu_*EUcP - EUc_muP
    ret[15] = alpha1 - m_*EUcP # bond pricing
    
    return ret
    
def G(w):
    '''
    Aggregate equations
    '''
    logm,muhat,e,c,l,rho1_,rho2,phi,w_e,UcP,a,I,x_,kappa_ = w[:ny] #y
    alpha1,alpha2,tau,eta,lamb,T = w[ny+ne:ny+ne+nY] #Y
    Eps = w[ny+ne+nY+nz+nv+n_p+neps] #aggregate shock
 
    
    Uc = c**(-sigma)
    
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
    logm,muhat,e,c,l,rho1_,rho2,phi,w_e,UcP,I,a,x_,kappa_ = y
    
    m = np.exp(logm)
    mu = muhat * m
    
    
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
    c = (Uc)**(-1/sigma)
    Ucc = -sigma*c**(-sigma-1)
    
    Ul = -(1-tau)*Uc*np.exp(e)
    l = ( -Ul/psi )**(1./gamma)
    Ull = -psi*gamma*l**(gamma-1)
    
    phi = ( Ul - mu*(Ull*l + Ul) + lamb*np.exp(e) )/Ull
    
    rho1 = (Uc - mu*(Ucc*c + Uc)- phi*np.exp(e)*(1-tau)*Ucc - lamb)/ (m*Ucc*(1-1./beta))
    rho2 = -rho1
    
    x = beta*( Uc*c + Ul*l )/(1-beta)
    
    kappa = rho1*Uc
    
    w_e = np.exp(e)   
    
    a = x/Uc
    
    I = a/beta +T    
    
    return np.vstack((
    logm,muhat,e,c,l,rho1,rho2,phi,w_e,Uc,a,I,x,kappa   
    ))
    
    

    
def GSS(YSS,y_i):
    '''
    Aggregate conditions for the steady state
    '''
    logm,muhat,e,c,l,rho1,rho2,phi,w_e,UcP,a,I,x,kappa = y_i
    alpha1,alpha2,tau,eta,lamb,T = YSS
  
    
    Uc = c**(-sigma)
    
    return np.hstack((
    alpha1-alpha2, np.mean(c-l*w_e),eta,np.mean(phi*Uc*w_e),np.mean(rho1),T    
    ))
    
def time0Finv(YSS,z):
    '''
    Given steady state YSS solves for y_i
    '''
    logm,e = z
    alpha1,alpha2,tau,eta,lamb,T = YSS

    m = np.exp(logm)    
    w_e = np.exp(e)
        
    Uc =alpha1/m
    c = (Uc)**(-1/sigma)
    Ucc = -sigma*c**(-sigma-1)
    
    Ul = -(1-tau)*Uc*w_e
    l = ( -Ul/psi )**(1./gamma)
    Ull = -psi*gamma*l**(gamma-1)
    
    mu = ( Uc-lamb - w_e*(1-tau)*Ucc*(Ul+lamb*w_e)/Ull )/( Ucc*(c-T)+Uc - w_e*(1-tau)*Ucc*(l+Ul/Ull) )

    phi = (Ul+lamb*w_e)/Ull - (l+Ul/Ull)*mu    
    
    muhat = mu/m
    
    x = (Uc*(c-T)+Ul*l)*beta/(1-beta)
    a = x/(beta*Uc)
    y = w_e * l
    
    test1 = Ul-mu*(Ull*l+Ul)   -phi *Ull +lamb*w_e
    test2 = Uc - mu*(Ucc*(c-T)+Uc) - phi*w_e*(1-tau)*Ucc - lamb
    
    return np.vstack((
    logm,muhat,e,c,l,phi,w_e,Uc,a,x,y,test1,test2
    ))
    
def time0GSS(YSS,y_i):
    '''
    Aggregate conditions for the steady state
    '''
    logm,muhat,e,c,l,phi,w_e,Uc,a,x,y,test1,test2 = y_i
    alpha1,alpha2,tau,eta,lamb,T = YSS
  
    
    Uc = c**(-sigma)
    
    return np.hstack((
    alpha1-alpha2, np.mean(c-l*w_e),eta,np.mean(phi*Uc*w_e),np.mean(muhat),T   
    ))
    
def nomalize(Gamma,weights =None):
    '''
    Normalizes the distriubtion of states if need be
    '''
    if weights == None:            
        Gamma[:,0] -= np.mean(Gamma[:,0])
        Gamma[:,1] -= np.mean(Gamma[:,1]) #/np.exp(Gamma[:,0]))*np.exp(Gamma[:,0])
    else:
        Gamma[:,0] -= weights.dot(Gamma[:,0])
        Gamma[:,1] -= weights.dot(Gamma[:,1])
    return Gamma
    
    
    