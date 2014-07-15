# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 12:18:27 2014

@author: dgevans
"""
import numpy as np
import pycppad as ad

beta = 0.95
gamma = 1.
sigma = 2.
sigma_e = np.array([0.06,0.2])
sigma_E = 0.03
mu_e = 0.7
psi = 10.
delta = 0.05
xi_l = 0.4
xi_k = 0.2

Gov = 0.17


tau_l = 0.3
tau_k = 0.15
T = 0.2


n = 2 # number of measurability constraints
nG = 1 # number of aggregate measurability constraints.
ny = 11 # number of individual controls (m_{t},mu_{t},c_{t},l_{t},rho1_,rho2,phi,x_{t-1},kappa_{t-1}) Note that the forward looking terms are at the end
ne = 4 # number of Expectation Terms (E_t u_{c,t+1}, E_t u_{c,t+1}mu_{t+1} E_{t}x_{t-1 E_t rho_{1,t-1}} [This makes the control x_{t-1},rho_{t-1} indeed time t-1 measuable])
nY = 4 # Number of aggregates (alpha_1,alpha_2,tau,eta,lambda)
nz = 2 # Number of individual states (m_{t-1},mu_{t-1})
nv = 2 # number of forward looking terms (x_t,rho1_t)
n_p = 1 #number of parameters
nZ = 1 # number of aggregate states
neps = len(sigma_e)

phat = np.array([-0.01])

indx_y={'logm':0,'muhat':1,'e':2,'c' :3,'l':4,'rho1_':5,'rho2':6,'phi':7,'wages':8,'UcP':9,'a':10,'x_':11,'kappa_':12,'pers_shock':13,'trans_shock':14}
indx_Y={'alpha1':0,'alpha2':1,'taxes':2,'eta':3,'lambda':4,'T':5,'shock':6}
indx_Gamma={'m_':0,'mu_':1,'e_':2}


def F(w):
    '''
    Individual first order conditions
    '''
    logm,a,c,l,k_,nl,r,pi,alpha1,f,x_,alpha1_ = w[:ny] #y
    EUc,EUc_r,Ex_,Ek_ = w[ny:ny+ne] #e
    K,Alpha1_,Alpha2,W = w[ny+ne:ny+ne+nY] #Y
    logm_,a_= w[ny+ne+nY:ny+ne+nY+nz] #z
    x,alpha1 = w[ny+ne+nY+nz:ny+ne+nY+nz+nv] #v
    nu = w[ny+ne+nY+nz+nv+n_p-1] #v
    K_ = w[ny+ne+nY+nz+nv+n_p+nZ-1] #Z
    eps_p,eps_t = w[ny+ne+nY+nz+nv+n_p+nZ:ny+ne+nY+nz+nv+n_p+nZ+neps] #shock
    Eps = w[ny+ne+nY+nz+nv+n_p+nZ+neps] #aggregate shock
    
    if (ad.value(logm) < -3.5):
        shock = 0.
    else:
        shock = 1.
    
    m_,m = np.exp(logm_),np.exp(logm)
    
    Uc = c**(-sigma)
    Ul = -psi*l**(gamma)
    
    ret = np.empty(ny+n,dtype=w.dtype)
    ret[0] = x_ - Ex_ # x is independent of eps
    ret[1] = k_ - Ek_
    
    ret[2] = Uc*(x_- alpha1_ * k_ / m_)/(beta*EUc) + (1-tau_k)*Uc*pi - Uc*(c-T) - Ul*l - x #impl
    ret[3] = alpha1 - m*Uc #defines m
    ret[4] = (1-tau_l)*W*Uc + Ul #wage
    ret[5] = r - np.exp(a) * xi_k * k_**(xi_k-1) * nl**xi_l - (1-delta)
    ret[6] = W - np.exp(a) * xi_l * k_**(xi_k) * nl**(xi_l-1)
    ret[7] = pi - ( np.exp(a)*k_**(xi_k)*nl**(xi_l) + (1-delta)*k_ - W*nl  )
    ret[8] = Alpha1_ - alpha1_
    #ret[9] = alpha1 - alpha1prime
    ret[9] = a - ( shock*(nu*a_+eps_p + (1-nu)*mu_e) + (1-shock)*a_  )
    ret[10] = f - np.exp(a) * xi_l * k_**(xi_k) * nl**(xi_l) - (1-delta)*k_ 
    
    ret[11] = Alpha2 - m_*EUc
    ret[12] = Alpha1_ - beta*(1-tau_k)*m_*EUc_r
    
    return ret
    
def G(w):
    '''
    Aggregate equations
    '''
    logm,a,c,l,k_,nl,r,pi,alpha1,f,x_,alpha1_ = w[:ny] #y
    K,Alpha1_,Alpha2,W = w[ny+ne:ny+ne+nY] #Y
    K_ = w[ny+ne+nY+nz+nv+n_p+nZ-1] #Z
    Eps = w[ny+ne+nY+nz+nv+n_p+nZ+neps] #aggregate shock
 
    
    ret = np.empty(nY+nG,dtype=w.dtype)
    ret[0] = logm 
    
    ret[1] = K_ - k_
    ret[2] = f - c - Gov - K
    ret[3] = l - nl
    
    ret[4] = logm
    
    return ret
    
def f(y):
    '''
    Expectational equations that define e=Ef(y)
    '''
    logm,a,c,l,k_,nl,r,pi,alpha1,f,x_,alpha1_ = y #y
    
    
    Uc = c**(-sigma)
    
    ret = np.empty(ne,dtype=y.dtype)
    ret[0] = Uc
    ret[1] = Uc * r
    ret[2] = x_
    ret[3] = k_
    
    return ret
    
def Finv(YSS,z):
    '''
    Given steady state YSS solves for y_i
    '''
    logm,a = z
    K,Alpha1_,Alpha1,Alpha2,W = YSS
    
    m = np.exp(logm)
    
    Uc =Alpha1/m
    c = (Uc)**(-1/sigma)
    
    Ul = -(1-tau_l)*Uc*W
    l = ( -Ul/psi )**(1./gamma)
    
    r = 1./(beta*(1-tau_k)) * np.ones(c.shape)
    A = np.exp(a)
    
    temp = A*xi_k*(W/(A*xi_l))**((xi_k-1)/xi_k)
    temp2 = xi_l + (xi_k-1)*(1-xi_l)/xi_k
    
    nl = ( (r-1+delta)/temp )**(1/temp2)
    k_ = (W/(xi_l*A))**(1/xi_k) * nl**((1-xi_l)/xi_k)
    alpha1 = Alpha1*np.ones(c.shape)
    alpha1_ = alpha1
    
    f = A*k_**(xi_k)*nl**(xi_l) + (1-delta) * k_
    pi = f - W*nl
    
    x_ = ( Uc*(c-T) + Ul*l - (1-tau_k)*(pi-r*k_) )/(1/beta-1)
    
    
    return np.vstack((
    logm,a,c,l,k_,nl,r,pi,alpha1,f,x_,alpha1_   
    ))
    
    

    
def GSS(YSS,y_i):
    '''
    Aggregate conditions for the steady state
    '''
    logm,a,c,l,k_,nl,r,pi,alpha1,f,x_,alpha1_ = y_i
    K,Alpha1_,Alpha1,Alpha2,W = YSS
  
    
    return np.hstack((
    Alpha1_-Alpha1,Alpha1 - Alpha2, np.mean(k_-K), np.mean(f - c - Gov - K), np.mean(l-nl)    
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
    else:
        Gamma[:,0] -= weights.dot(Gamma[:,0])
    return Gamma
    
    
    