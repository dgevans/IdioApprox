# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 12:18:27 2014

@author: dgevans
"""
import numpy as np
import pycppad as ad

beta = 0.95
gamma = 2.
sigma = 2.
sigma_e = np.array([0.01,0.1])
sigma_E = 0.03
mu_e = 0.
psi = 10.
delta = 0.05
xi_l = 0.7 *.7
xi_k = 0.3 *.7

Gov = 0.17



n = 4 # number of measurability constraints
nG = 1 # number of aggregate measurability constraints.
ny = 26 # number of individual controls (m_{t},mu_{t},c_{t},l_{t},rho1_,rho2,phi,x_{t-1},kappa_{t-1}) Note that the forward looking terms are at the end
ne = 10 # number of Expectation Terms (E_t u_{c,t+1}, E_t u_{c,t+1}mu_{t+1} E_{t}x_{t-1 E_t rho_{1,t-1}} [This makes the control x_{t-1},rho_{t-1} indeed time t-1 measuable])
nY = 10 # Number of aggregates (alpha_1,alpha_2,tau,eta,lambda)
nz = 3 # Number of individual states (m_{t-1},mu_{t-1})
nv = 8 # number of forward looking terms (x_t,rho1_t)
n_p = 1 #number of parameters
nZ = 1 # number of aggregate states
neps = len(sigma_e)

phat = np.array([-0.005])

indx_y={'logm':0,'muhat':1,'e':2,'c' :3,'l':4,'rho1_':5,'rho2':6,'phi':7,'wages':8,'UcP':9,'a':10,'x_':11,'kappa_':12,'pers_shock':13,'trans_shock':14}
indx_Y={'alpha1':0,'alpha2':1,'taxes':2,'eta':3,'lambda':4,'T':5,'shock':6}
indx_Gamma={'m_':0,'mu_':1,'e_':2}


def F(w):
    '''
    Individual first order conditions
    '''
    logm,muhat,a,c,l,k_,nl,r,pi,f,rho1,phi1,phi2,foc_k,foc_alpha1,alpha2_,foc_alpha2,foc_K,x_,alpha1_,rho2_,rho3_,foc_m_,kappa_,temp1_,temp2_ = w[:ny] #y
    EUc,EUc_r,Ex_,Ek_,EUc_mu,Erho2_,Erho3_,Efoc_k,Etemp1_,Etemp2_ = w[ny:ny+ne] #e
    K,Alpha1_,Alpha2_,W,tau_l,tau_k,Xi,Eta,Kappa_,Iota = w[ny+ne:ny+ne+nY] #Y
    logm_,muhat_,a_= w[ny+ne+nY:ny+ne+nY+nz] #z
    x,alpha1,rho2,rho3,foc_m,kappa,temp1,temp2 = w[ny+ne+nY+nz:ny+ne+nY+nz+nv] #v
    nu = w[ny+ne+nY+nz+nv+n_p-1] #v
    K_ = w[ny+ne+nY+nz+nv+n_p+nZ-1] #Z
    eps_p,eps_t = w[ny+ne+nY+nz+nv+n_p+nZ:ny+ne+nY+nz+nv+n_p+nZ+neps] #shock
    Eps = w[ny+ne+nY+nz+nv+n_p+nZ+neps] #aggregate shock
    
    #if (ad.value(logm) < -3.5):
    #    shock = 0.
    #else:
    #    shock = 1.
    shock = 1.
    m_,m = np.exp(logm_),np.exp(logm)
    mu_,mu = muhat_*m_,muhat*m
    
    Uc = c**(-sigma)
    Ucc = -sigma*c**(-sigma-1)
    Ul = -psi*l**(gamma)
    Ull = -psi*gamma*l**(gamma-1)
    A = np.exp(a+eps_t)    
    
    r_k = A * xi_k*(xi_k-1) * k_**(xi_k-2) * nl**xi_l
    fnk = A * xi_k * xi_l * k_**(xi_k-1) * nl**(xi_l-1)
    fnn = A  * xi_l*(xi_l-1) * k_**(xi_k) * nl**(xi_l-2)
    fn = A * xi_l * k_**(xi_k) * nl**(xi_l-1)
    
    ret = np.empty(ny+n,dtype=w.dtype)
    ret[0] = x_ - Ex_ # x is independent of eps
    ret[1] = k_ - Ek_
    ret[2] = rho2_ - Erho2_
    ret[3] = rho3_ - Erho3_
    
    ret[4] = Uc*x_/(beta*EUc) + Uc*((1-tau_k)*pi - Alpha1_*k_/(beta*Alpha2_)) - Uc*c - Ul*l - x #x
    ret[5] = alpha1 - m*Uc #rho1
    ret[6] = (1-tau_l)*W*Uc + Ul #phi1
    ret[7] = r - A * xi_k * k_**(xi_k-1) * nl**xi_l - (1-delta) #r
    ret[8] = W - fn #phi2
    ret[9] = pi - ( A*k_**(xi_k)*nl**(xi_l) + (1-delta)*k_ - W*nl  ) #pi
    ret[10] = Alpha1_ - alpha1_ #alpha1_
    ret[11] = a - ( shock*(nu*a_+eps_p + (1-nu)*mu_e) + (1-shock)*a_  ) #a
    ret[12] = f - A * k_**(xi_k) * nl**(xi_l) - (1-delta)*k_ #f  
    ret[13] = Uc + x*Ucc/(beta*EUc) *(mu-mu_) + mu*( Ucc*((1-tau_k)*pi - Alpha1_/(beta*Alpha2_)*k_)
              -Ucc*c - Uc ) - m*Ucc*(rho1 + rho2_/beta + rho3_ * r *(1-tau_k) ) - phi1 *(1-tau_l)*W*Ucc - Xi #c
    ret[14] = Ul - mu*(Ull*l + Ul) - phi1*Ull - Eta #l
    ret[15] = foc_k - (mu*Uc*((1-tau_k)*r - Alpha1_/(beta*Alpha2_)) - rho3_*m_*Uc*r_k*(1-tau_k) - 
                phi2*fnk - Kappa_/beta + r* Xi ) #fock
    ret[16] = -phi2*fnn + Eta + Xi*fn #nl
    ret[17] = foc_m_ - (rho2_*EUc + rho3_ *(1-tau_k)*beta*EUc_r) #foc_m_
    ret[18] = rho1*Uc + foc_m + Iota/m #logm
    ret[19] = Kappa_ - kappa_ #kappa_
    ret[20] = temp1_ - Etemp1_ #temp1
    ret[21] = temp2_ - Etemp2_ #temp2
    ret[22] = foc_alpha1 - (rho1 + rho3 + temp1) #focalpha1
    ret[23] = foc_alpha2 - (rho2 + temp2) #foc_alpha2
    ret[24] = foc_K - (kappa-Xi) #foc_K
    ret[25] = alpha2_ - Alpha2_
    
    ret[26] = Efoc_k #k_
    ret[27] = mu_*EUc - EUc_mu #muhat
    ret[28] = Alpha2_ - m_*EUc #rho2_
    ret[29] = Alpha1_ - beta*(1-tau_k)*m_*EUc_r #rho3
    
    return ret
    
def G(w):
    '''
    Aggregate equations
    '''
    logm,muhat,a,c,l,k_,nl,r,pi,f,rho1,phi1,phi2,foc_k,foc_alpha1,alpha2_,foc_alpha2,foc_K,x_,alpha1_,rho2_,rho3_,foc_m_,kappa_,temp1_,temp2_ = w[:ny] #y
    logm_,muhat_,a_= w[ny+ne+nY:ny+ne+nY+nz] #z
    K,Alpha1_,Alpha2_,W,tau_l,tau_k,Xi,Eta,Kappa_,Iota = w[ny+ne:ny+ne+nY] #Y
    K_ = w[ny+ne+nY+nz+nv+n_p+nZ-1] #Z
    Eps = w[ny+ne+nY+nz+nv+n_p+nZ+neps] #aggregate shock
    
    m_,m = np.exp(logm_),np.exp(logm)
    mu_,mu =muhat_*m_,muhat*m
    Uc = c**(-sigma)
    
 
    
    ret = np.empty(nY+nG,dtype=w.dtype)
    ret[0] = logm 
    
    ret[1] = K_ - k_
    ret[2] = f - c - Gov - K
    ret[3] = l - nl
    ret[4] = rho3_*m_*Uc*r - mu*Uc*pi
    ret[5] = phi1*Uc
    ret[6] = foc_alpha1
    ret[7] = foc_alpha2
    ret[8] = phi2 - phi1*(1-tau_l)*Uc
    ret[9] = foc_K
    
    ret[10] = logm
    
    return ret
    
def f(y):
    '''
    Expectational equations that define e=Ef(y)
    '''
    logm,muhat,a,c,l,k_,nl,r,pi,f,rho1,phi1,phi2,foc_k,foc_alpha1,alpha2_,foc_alpha2,foc_K,x_,alpha1_,rho2_,rho3_,foc_m_,kappa_,temp1_,temp2_ = y #y
    
        
    m = np.exp(logm)
    mu = muhat*m
    Uc = c**(-sigma)
    
    ret = np.empty(ne,dtype=y.dtype)
    ret[0] = Uc
    ret[1] = Uc * r
    ret[2] = x_
    ret[3] = k_
    ret[4] = Uc*mu
    ret[5] = rho2_
    ret[6] = rho3_
    ret[7] = foc_k
    ret[8] = -mu*k_*Uc/alpha2_
    ret[9] = mu*alpha1_*k_*Uc/(alpha2_)**2    
        
    
    return ret
    
def Finv(YSS,z):
    '''
    Given steady state YSS solves for y_i
    '''
    logm,muhat,a = z
    K,Alpha1_,Alpha2_,W,tau_l,tau_k,Xi,Eta,Kappa_,Iota = YSS
    
    m = np.exp(logm)
    mu = muhat*m
    
    Uc =Alpha1_/m
    c = (Uc)**(-1/sigma)
    Ucc = -sigma * c**(-sigma-1)
    
    Ul = -(1-tau_l)*Uc*W
    l = ( -Ul/psi )**(1./gamma)
    Ull = -psi * gamma * l**(gamma-1)
    
    r = 1./(beta*(1-tau_k)) * np.ones(c.shape)
    A = np.exp(a)
    
    temp = A*xi_k*(W/(A*xi_l))**((xi_k-1)/xi_k)
    temp2 = xi_l + (xi_k-1)*(1-xi_l)/xi_k
    
    nl = ( (r-1+delta)/temp )**(1/temp2)
    k_ = (W/(xi_l*A))**(1/xi_k) * nl**((1-xi_l)/xi_k)
    alpha1 = Alpha1_*np.ones(c.shape)
    alpha1_ = alpha1
    
    f = A*k_**(xi_k)*nl**(xi_l) + (1-delta) * k_
    fn = xi_l *A*k_**(xi_k)*nl**(xi_l-1)
    fnn = xi_l*(xi_l-1)*A*k_**(xi_k)*nl**(xi_l-2)
    fk = xi_k * A*k_**(xi_k-1)*nl**(xi_l) + (1-delta)
    fkk = xi_k *(xi_k-1) * A*k_**(xi_k-2)*nl**(xi_l)
    fnk = xi_k * xi_l * A*k_**(xi_k-1)*nl**(xi_l-1)
    
    pi = f - W*nl
    
    x_ = ( Uc*c + Ul*l - Uc*((1-tau_k)*pi-k_/beta) )/(1/beta-1)
    
    phi1 = (Ul - mu*(Ull*l + Ul) - Eta)/Ull
    #Ul - mu*(Ull*l + Ul) - phi1*Ull - Eta
    phi2 = (Eta + Xi*fn)/fnn
    rho3 = (fk*Xi - Kappa_/beta - phi2*fnk)/(m*Uc*fkk*(1-tau_k))
    rho1 = ( Uc + mu*(Ucc*( (1-tau_k)*pi - k_/beta  ) - Ucc*c -Uc ) - phi1*(1-tau_l)*W*Ucc - Xi)/(Ucc*m*(1-1./beta))    
    rho2 = -(rho3+rho1)
    
    alpha2_ = alpha1
    temp1_ = -mu*k_*Uc/alpha2_
    temp2_ = mu*alpha1_*k_*Uc/(alpha2_)**2
    
    foc_alpha1 = rho1 + rho3 + temp1_
    foc_alpha2 = rho2 + temp2_
    kappa_ = Kappa_ * np.ones(c.shape)
    
    foc_k = (mu*Uc*((1-tau_k)*r - Alpha1_/(beta*Alpha2_)) - rho3*m*Uc*fkk*(1-tau_k) - 
                phi2*fnk - Kappa_/beta + r* Xi )
    foc_K = kappa_-Xi
    foc_m_ = rho2*Uc + rho3 *(1-tau_k)*beta*Uc*r
    
    return np.vstack((
    logm,muhat,a,c,l,k_,nl,r,pi,f,rho1,phi1,phi2,foc_k,foc_alpha1,alpha2_,foc_alpha2,foc_K,x_,alpha1_,rho2,rho3,foc_m_,kappa_,temp1_,temp2_   
    ))
    
    

    
def GSS(YSS,y_i):
    '''
    Aggregate conditions for the steady state
    '''
    logm,muhat,a,c,l,k_,nl,r,pi,f,rho1,phi1,phi2,foc_k,foc_alpha1,alpha2_,foc_alpha2,foc_K,x_,alpha1_,rho2,rho3,foc_m_,kappa_,temp1_,temp2_  = y_i
    K,Alpha1_,Alpha2_,W,tau_l,tau_k,Xi,Eta,Kappa_,Iota = YSS
  
    m = np.exp(logm)
    mu = m*muhat
    Uc = c**(-sigma)    
    
    return np.hstack((
    Kappa_-Xi,Alpha1_-Alpha2_,np.mean(rho3*m*Uc*r - mu*Uc*pi),np.mean(phi1*Uc),np.mean(-(mu*k_*Uc)/Alpha1_ +rho1 + rho3 ),np.mean(phi2-phi1*(1-tau_l)*Uc) , np.mean(k_-K), np.mean(f - c - Gov - K), np.mean(l-nl),Iota    
    ))
    

    
def nomalize(Gamma,weights =None):
    '''
    Normalizes the distriubtion of states if need be
    '''
    if weights == None:            
        Gamma[:,0] -= np.mean(Gamma[:,0])
        Gamma[:,1] -= np.mean(Gamma[:,1])
    else:
        Gamma[:,0] -= weights.dot(Gamma[:,0])
        Gamma[:,1] -= weights.dot(Gamma[:,1])
    return Gamma
    
    
    