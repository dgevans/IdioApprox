# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 19:49:53 2014

@author: dgevans
"""
import steadystate
import calibrate_begs_id_nu_ghh as Para
from scipy.optimize import root
import numpy as np

N = 10000

steadystate.calibrate(Para)
steadystate.Finv = Para.time0Finv
steadystate.GSS = Para.time0GSS

#match  moments

def f(z):
    mu = z[:2]
    Sigma = np.array([[z[2],z[3]],[z[3],z[4]]])
    Sig = np.diag(Sigma)
    mu_y,mu_a = np.exp(mu + 0.5*Sig)
    Sigma2 = np.exp(mu+mu.reshape(-1,1) + 0.5*(Sig+Sig.reshape(-1,1)))*(np.exp(Sigma)-1)
    sig_y = np.sqrt(Sigma2[0,0])
    sig_a = np.sqrt(Sigma2[1,1])
    corr_ya = Sigma2[0,1]/(sig_y*sig_a)
    return np.array([mu_y,mu_a,sig_y/mu_y,sig_a/mu_a,corr_ya]) - np.array([1.,5.,2.65,6.53,0.5])
    
z = root(f,np.array([1.,1,1.,0.,1.])).x
mu = z[:2]
Sigma = np.array([[z[2],z[3]],[z[3],z[4]]])
Gamma_ya = np.exp(np.random.multivariate_normal(mu,Sigma,N))
beta,sigma,gamma,psi = Para.beta,Para.sigma,Para.gamma,Para.psi
T,tau = 0.2,0.3
y,a = Gamma_ya.T
c = (1-beta)*a/beta + T + (1-tau)*y

l = (psi*(1-tau)*y)**(1/(1+gamma))
e = np.log(y/l)
chat = c-psi*l**(1+gamma)/(1+gamma)
Uc = chat**(-sigma)

logm = -np.log(Uc)
logm -= logm.mean()
Gamma0= np.vstack([logm,e]).T


        
ss = steadystate.steadystate(Gamma0.T)

Gamma,Y,Shocks,y = {},{},{},{}
Gamma[0] = ss.get_y(ss.z_i)[:3,:].T

import simulate
v = simulate.v
v.execute('import calibrate_begs_id_nu_ghh as Para')
v.execute('import approximate_begs as approximate')
v.execute('approximate.calibrate(Para)')
v.execute('approximate.shock = 0.')
simulate.simulate(Para,Gamma,Y,Shocks,y,100) #simulate 150 period with no aggregate shocks
Gamma0 = Gamma[9]
#Simulate all high shocks
v.execute('approximate.shock = 1.')
v.execute('import numpy as np')
v.execute('state = np.random.get_state()')
Gamma,Y,Shocks,y = {},{},{},{}
Gamma[0] = Gamma0
simulate.simulate(Para,Gamma,Y,Shocks,y,50)
YH = np.vstack(Y.values())
#simulate all no shocks
v.execute('approximate.shock = 0.')
v.execute('np.random.set_state(state)')
Gamma,Y,Shocks,y = {},{},{},{}
Gamma[0] = Gamma0
simulate.simulate(Para,Gamma,Y,Shocks,y,50)
Y0 = np.vstack(Y.values())
#simulate all low shocks
v.execute('approximate.shock = -1.')
v.execute('np.random.set_state(state)')
Gamma,Y,Shocks,y = {},{},{},{}
Gamma[0] = Gamma0
simulate.simulate(Para,Gamma,Y,Shocks,y,50)
YL = np.vstack(Y.values())