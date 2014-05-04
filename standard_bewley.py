# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 21:15:35 2014

@author: anmol
"""

from numpy import *
from scipy import stats
from Spline import Spline
from pylab import *

class Bewley: 


    def  __init__(self):
        self.beta = 0.95
        self.sigma = 1.
        self.gamma = 2.
        self.sigma_e = 0.5
        self.mu_y=1
        self.T = 0
        self.tau = 0        
        self.n_y=10
        self.n_a=15
        self.u_c=lambda c: c**-self.gamma
        self.inv_u_c=lambda uc: uc**(-1/self.gamma)
        self.R=(1/self.beta)*.98
        self.eps=.0001
        
    
    def build_grid(self):
        
        # define a grid on income and assets
        self.y_grid=self.mu_y+linspace(-1.95,1.95,self.n_y)*self.sigma_e
        self.p=stats.norm.pdf(self.y_grid,self.mu_y,self.sigma_e)
        self.p=self.p/sum(self.p)

        self.a_min=-(1/(1-self.beta))*min(self.y_grid)*0
        self.a_max= (1/(1-self.beta))*max(self.y_grid)
        self.agrid= hstack((linspace(self.a_min,self.a_min +.1,10),linspace(self.a_min+0.1,1.,3)[1:],linspace(1.,self.a_max,2)[1:]))
        self.n_a=len(self.agrid)        
        self.domain=zeros((self.n_a*self.n_y,2))
        n=0
        for ind_y in range(self.n_y):
            for ind_a in range(self.n_a):   
                self.domain[n,:]= ([self.agrid[ind_a],self.y_grid[ind_y]])
                n=n+1               
        self.c_policy= Spline(self.domain,self.domain[:,1],([1,1]))
        self.endogenous_grid=self.domain 
               
         
    def c(self,state):
        
        a=state[0]
        y=state[1]
        end_grid_for_y=self.endogenous_grid[where(self.endogenous_grid[:,1]==y)]
        flag_borrowin_constraint=0        
        if a< min(end_grid_for_y[:,0]):
            flag_borrowin_constraint=1            
        
        return maximum(self.c_policy(state)*(1-flag_borrowin_constraint)+flag_borrowin_constraint*(a*self.R+y-self.a_min),self.eps)
        
         
    def c_tilde(self,state):
        states_tomorrow=hstack((ones((self.n_y,1))*state[0],self.y_grid.reshape(self.n_y,1)))
        c_tomorrow=array(map(self.c,states_tomorrow))
        return self.inv_u_c(self.beta*self.R*self.u_c(c_tomorrow.T).dot(self.p))
           
    def draw_shock(self,values, probabilities, size):
        bins = np.add.accumulate(probabilities)
        return values[np.digitize(random_sample(size), bins)]
    
        
    def update_c(self):        
        c_tilde = array(map(self.c_tilde,self.domain)).flatten()
        end_grid_assets=(self.domain[:,0]+c_tilde-self.domain[:,1])/self.R
        end_grid=vstack((end_grid_assets,self.domain[:,1])).T
        self.endogenous_grid=end_grid                  
        self.c_policy=Spline(end_grid,c_tilde ,([1,1]))        
        
    def simulate(self,T,a0,y0):
        c_hist=zeros((T,1))
        a_hist=zeros((T+1,1))
        y_hist=zeros((T+1,1))
        a_hist[0]=a0
        y_hist[0]=y0
        
        for t in range(T):
            c_hist[t]=self.c([a_hist[t],y_hist[t]])
            a_hist[t+1]=a_hist[1]*self.R+y_hist[t]-c_hist[t]
            y_hist[t+1]=self.draw_shock(self.y_grid,self.p,1)
            
        return c_hist, a_hist, y_hist 
            
    
    

err=0
b=Bewley()
b.build_grid()    
b.update_c()    



for n in range(200):
    b.update_c()    
    diff=err-max(b.c_policy.getCoeffs())
    err=max(b.c_policy.getCoeffs())
    print diff
#hist, a_hist, y_hist =b.simulate(2000,1,1)
#hist(a_hist)
#show()
plot_domain=lambda y: concatenate((b.agrid.reshape(b.n_a,1),ones((b.n_a,1))*y),axis=1)
plot(b.agrid,b.domain[0,:]*(b.R-1)-map(b.c,plot_domain(b.y_grid[3]))+b.y_grid[3])
show()