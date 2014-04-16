# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 17:28:52 2014

@author: dgevans
"""
from copy import copy
import numdifftools as nd
import steadystate
import numpy as np
from Spline import Spline
import utilities
from utilities import hashable_array
from utilities import quadratic_dot
import itertools

F =None
G = None
f = None
h = None

n = None
ny = None
ne = None
nY = None
nz = None
nv = None

y,e,Y,z,v,eps,S = (None)*7
interpolate = utilities.interpolator_factory([3])

def calibrate(Para):
    global F,G,f,h,ny,ne,nY,nz,nv,n
    global y,e,Y,z,v,eps,S
    global interpolate
    F,G,f,h = Para.F,Para.G,Para.f,Para.h
    ny,ne,nY,nz,nv,n = Para.ny,Para.ne,Para.nY,Para.nz,Para.nv,Para.n
    
    #store the indexes of the various types of variables
    y = np.arange(ny).view(hashable_array)
    e = np.arange(ny,ny+ne).view(hashable_array)
    Y = np.arange(ny+ne,ny+ne+nY).view(hashable_array)
    z = np.arange(ny+ne+nY,ny+ne+nY+nz).view(hashable_array)
    v = np.arange(ny+ne+nY+nz,ny+ne+nY+nz+nv).view(hashable_array)
    eps = np.arange(ny+ne+nY+nz+nv,ny+ne+nY+nz+nv+1).view(hashable_array)
    
    S = np.hstack((z,Y)).view(hashable_array)
    
    interpolate = utilities.interpolator_factory([3]*nz)

class approximate(object):
    '''
    Computes the second order approximation 
    '''
    def __init__(self,Gamma):
        '''
        Approximate the equilibrium policies given z_i
        '''
        self.Gamma = Gamma
        self.ss = steadystate.steadystate(Gamma)
        self.zgrid =None
        
        #precompute Jacobians and Hessians
        self.DF = interpolate(self.zgrid,map(lambda z_i:nd.Jacobian(F)(self.get_w(z_i)),self.zgrid))
        self.HF = interpolate(self.zgrid,map(lambda z_i:utilities.Hessian(F,self.get_w(z_i)),self.zgrid))
        
        self.DG = interpolate(self.zgrid,map(lambda z_i:nd.Jacobian(G)(self.get_w(z_i)),self.zgrid))
        self.HG = interpolate(self.zgrid,map(lambda z_i:utilities.Hessian(G,self.get_w(z_i)),self.zgrid))
        
        self.df = interpolate(self.zgrid,map(lambda z_i:nd.Jacobian(f)(self.get_w(z_i)[y]),self.zgrid))
        self.Hf = interpolate(self.zgrid,map(lambda z_i:utilities.Hessian(f,self.get_w(z_i)[y]),self.zgrid))
        
        self.dh = interpolate(self.zgrid,map(lambda z_i:nd.Jacobian(h)(self.get_w(z_i)[e]),self.zgrid))
        self.Hh = interpolate(self.zgrid,map(lambda z_i:utilities.Hessian(h,self.get_w(z_i)[e]),self.zgrid))
    def integrate(self,f):
        '''
        Integrates a function f over Gamma
        '''
        return sum(itertools.imap(f,self.Gamma))/len(self.Gamma)
    
    def get_w(self,z_i):
        '''
        Gets w for particular z_i
        '''
        ybar = self.ss.get_y(z_i)
        Ybar = self.ss.get_Y()
        ebar = f(ybar)
        
        return np.hstack((
        ybar,ebar,Ybar,ybar[:nz],h(ebar),0.
        ))
        
    def compute_dy(self,z_i):
        '''
        Computes dy
        '''
        
        DF = self.DF(z_i)
        df = self.df(z_i)
        dh = self.dh(z_i)
        
        dy = {}
        DFi = DF[n:]
        dy[z] = np.linalg.solve(DFi[:,y] + DFi[:,e].dot(df)+DFi[:,v].dot(dh).dot(df),
                                    -DFi[:,z])
        DFi = DF[:-n,:]
        Izy = np.zeros(nz,ny)
        Izy[:nz,:nz] = np.eye(nz)
        dy[eps] = np.linalg.solve(DFi[:,y] + DFi[:,v].dot(dh).dot(df).dot(dy[z]).dot(Izy),
                                    -DFi[:,eps])
                                    
        DFi = DF[n:,:]
        dy[Y] = np.linalg.solve(DFi[:,y] + DFi[:,e].dot(df) + DF[:,v].dot(dh).dot(df),
                                -DFi[:,Y])
                                
        return dy
        
    def linearize(self):
        '''
        Computes the linearization
        '''
        self.dy = {}
        temp = utilities.dict_map(self.compute_dy,self.zgrid)
        for x in [z,eps,Y]:
            self.dy[x] = interpolate(self.zgrid,temp[x])
        
        DG = self.DG
        
        Ghat = self.integrate(lambda z_i: DG[:,Y](z_i)+DG[:,y](z_i).dot(self.dy[Y](z_i)))
        
        self.dY = interpolate(self.zgrid,
                              map(lambda z_i: np.linalg.solve(Ghat,
                            -DG[:,z](z_i)-DG[:,y](z_i).dot(self.dy[z](z_i))),self.zgrid))
                            
    def compute_d2y(self,z_i):
        '''
        Computes second derivative of y
        '''
        DF,HF = self.DF(z_i),self.HF(z_i)
        df,Hf = self.df(z_i),self.Hf(z_i)
        dh,Hh = self.dh(z_i),self.Hh(z_i)
        dy = self.dy
        
        #first compute DFhat, need loadings of S and epsilon on variables
        Izy = np.zeros(nz,ny)
        Izy[:nz,:nz] = np.eye(nz)
        d = {}
        d[y,S],d[y,eps] = np.hstack((self.dy[z](z_i),self.dy[Y](z_i))),self.dy[eps](z_i)
    
        d[e,S],d[e,eps] = df.dot(d[y][S]), np.zeros((ne,1))
    
        d[Y,S],d[Y,eps] = np.hstack(( np.zeros((nY,nz)), np.eye(nY) )),np.zeros((nY,1))
        
        d[z,S],d[z,eps] = np.hstack(( np.eye(nz), np.zeros((nz,nY)) )),np.zeros((nz,1))
        
        d[v,S],d[v,eps] = dh.dot(d[e][S]), dh.dot(df).dot(self.dy[z](z_i)).dot(Izy).dot(self.dy[eps](z_i))
        
        d[eps,S],d[eps,eps] = np.zeros((1,nz+nY)),np.eye(1)
        
        #Now compute Fhat
        HFhat = {}
        HFhat[S,S],HFhat[S,eps],HFhat[eps,S],HFhat[eps,eps] = 0.,0.,0.,0.
        for x1 in [y,e,Y,z,v,eps]:
            for x2 in [y,e,Y,z,v,eps]:
                HFhat[S,S] += quadratic_dot(HF[n:,x1,x2],d[x1,S],d[x2,S])
                HFhat[S,eps] += quadratic_dot(HF[:-n,x1,x2],d[x1,S],d[x2,eps])
                HFhat[eps,S] += quadratic_dot(HF[:-n,x1,x2],d[x1,eps],d[x2,S])
                HFhat[eps,eps] += quadratic_dot(HF[:-n,x1,x2],d[x1,eps],d[x2,eps])
                
        #Now compute d2y
        d2y = {}
        d2y[S,S] = np.tensordot(np.linalg.inv(DF[n:,y] + DF[n:,e].dot(df) + DF[n:,v].dot(dh).dot(df))
        -HFhat[S,S] - np.tensordot(DF[n:,e],quadratic_dot(Hf,d[y,S],d[y,S]),1)
        -np.tensordot( DF[n:,v], np.tensordot(dh,quadratic_dot(Hf,d[y,S],d[y,S]),1)
        +quadratic_dot(Hh,df.dot(d[y,S]),df.dot(d[y,S])),1 ), axis=1        
        )
        
        d[y,z] = d[y,S][:,:nz]
        d[z,eps] = Izy.dot(d[y,eps])
        DFi = DF[:-n]
        d2y[S,eps] =np.tensordot(np.linalg.inv(DFi[:,y] + DFi[:,v].dot(dh).dot(df).dot(dy[z]).dot(Izy)),
        -HFhat[S,eps] - np.tensordot(DFi[:,v], np.tensordot(dh.dot(df), d2y[S,S][:,:,:nz].dot(d[z,eps]),1)
        +np.tensordot(dh,quadratic_dot(Hf,d[y,S],d[y,z].dot(d[z,eps])),1)
        +quadratic_dot(Hh,df.dot(d[y,S]),df.dot(d[y,z]).dot(d[z,eps])),1)                   
        , axis =1 )
        d2y[eps,S] = copy(d2y[S,eps])
        d2y[eps,S].transpose(0,2,1)
        
        d2y[eps,eps] = np.tensordot(np.linalg.inv(DFi[:,y] + DFi[:,v].dot(dh).dot(df).dot(dy[z]).dot(Izy)),
        -HFhat[eps,eps] - np.tensordot(DFi[:,v],
        np.tensordot(dh.dot(df), quadratic_dot(d2y[S,S][:,:nz,:nz],d[z,eps],d[z,eps]),1)
        +np.tensordot(dh,quadratic_dot(Hf,d[y,z].dot(d[z,eps]),d[y,z].dot(d[z,eps])),1)
        +quadratic_dot(Hh,df.dot(d[y,z]).dot(d[z,eps]),df.dot(d[y,z]).dot(d[z,eps]),1)
        ,axis=1)
        ,axis=1)
        
        return d2y
        
    def compute_d2Y(self,z_i):
        '''
        Computes the second order approximation for agent of type z_i
        '''
        DG,HG = self.DG(z_i),self.HG(z_i)
        
            