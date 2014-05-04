# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 17:26:22 2014

@author: dgevans
"""

import numpy as np
from SparseGrid import interpolator
from Spline import Spline
import pycppad as ad

def hstack(tup):
    '''
    hstack for interpolate wrapper
    '''
    N = tup[0].N
    tup_f = map(lambda t: t.F, tup)
    return interpolate_wrapper(np.hstack(tup_f),N)

def vstack(tup):
    '''
    hstack for interpolate wrapper
    '''
    N = tup[0].N
    tup_f = map(lambda t: t.F, tup)
    return interpolate_wrapper(np.vstack(tup_f),N)

def dict_map(F,l):
    '''
    perform a map preserving the dict structure
    '''
    ret = {}
    temp = map(F,l)
    keys = temp[0].keys()
    for key in keys:
        ret[key] = [t[key] for t in temp]
    return ret
    
class interpolate_wrapper(object):
    '''
    Wrapper to interpolate vector function
    '''
    def __init__(self,F,N):
        '''
        Inits with array of interpolated functions
        '''
        self.F = F
        self.N = N
    def __getitem__(self,index):
        '''
        Uses square brakets operator
        '''
        return interpolate_wrapper(self.F[index],self.N)
    def reshape(self,*args):
        '''
        Reshapes F
        '''
        newF = self.F.reshape(*args)
        return interpolate_wrapper(newF,self.N)
        
    def transpose(self,*axes):
        '''
        Creates transpolse
        '''
        newF = self.F.transpose(*axes)
        return interpolate_wrapper(newF,self.N)
    
    def __len__(self):
        '''
        return length
        '''
        return len(self.F)
    def __call__(self,X):
        '''
        Evaluates F at X for each element of F, keeping track of the shape of F
        '''
        X = np.atleast_1d(X)
        if self.N == 1:
            if len(X) == 1:
                fhat = np.vectorize(lambda f: float(f(X)),otypes=[np.ndarray])
            else:
                fhat = np.vectorize(lambda f: f(X).flatten(),otypes=[np.ndarray])
            return np.array(fhat(self.F).tolist())
        else:
            if X.ndim == 1 or len(X) == 1:
                fhat = np.vectorize(lambda f: float(f(X)),otypes=[np.ndarray])
            else:
                fhat = np.vectorize(lambda f: f(X).flatten(),otypes=[np.ndarray])
            return np.array(fhat(self.F).tolist())

class interpolator_factory(object):
    '''
    Generates an interpolator factory which will interpolate vector functions
    '''
    def __init__(self,d,mu,Mu,Sigma):
        '''
        Inits with types, orders and k
        '''
        self.interpolate = interpolator(d,mu,Mu,Sigma)
        self.X = self.interpolate.getX()
        
    def __call__(self,Fs):
        '''
        Interpolates function given function values Fs at domain X
        '''
        Fshape = Fs[0].shape
        Fflat = np.vstack(map(lambda F:F.flatten(),Fs))
        F = []
        for F_i in Fflat.T:
            F.append(self.interpolate.fit(F_i))
        return interpolate_wrapper(np.array(F),self.X.shape[1]).reshape(Fshape)

class interpolator_factory_spline(object):
    '''
    Generates an interpolator factory which will interpolate vector functions
    '''
    def __init__(self,k):
        '''
        Inits with types, orders and k
        '''
        self.k = k
        
    def __call__(self,X,Fs):
        '''
        Interpolates function given function values Fs at domain X
        '''
        Fshape = Fs[0].shape
        Fflat = np.vstack(map(lambda F:F.flatten(),Fs))
        F = []
        for F_i in Fflat.T:
            F.append(Spline(X,F_i,self.k))
        if X.ndim == 1:
            return interpolate_wrapper(np.array(F),1).reshape(Fshape)
        else:
            return interpolate_wrapper(np.array(F),np.atleast_2d(X).shape[1]).reshape(Fshape)
            
def nd_Hessian(F,x0):
    '''
    Computes the hessian of F
    '''
    n = x0.shape[0]
    m = np.asarray(F(x0)).shape[0]
    HF = np.empty((m,n,n))
    for i in range(m):
        HF[i] = nd.Hessian(lambda x:F(x)[i])(x0)
    
    return HF
    
    
def ad_Jacobian(F,x0):
    '''
    Computes Jacobian using automatic differentiation
    '''
    a_x = ad.independent(x0)
    a_F = F(a_x)
    return ad.adfun(a_x,a_F).jacobian(x0)

def ad_Hessian(F,x0):
    '''
    Computes Hessian of F using automatic differentiation
    '''
    a_x = ad.independent(x0)
    a_F = F(a_x)
    n = x0.shape[0]
    m = a_F.shape[0]
    HF = np.empty((m,n,n))
    I = np.eye(m)
    
    adF = ad.adfun(a_x,a_F)
    for i in range(m):
        HF[i,:,:] = adF.hessian(x0,I[i])
    
    return HF

from hashlib import sha1

from numpy import all, array


class hashable_array(np.ndarray):
    __hash = None
    def __new__(cls, values):
        this = np.ndarray.__new__(cls, shape=values.shape, dtype=values.dtype)
        this[...] = values
        return this
    
    def __init__(self, values):
        self.__hash = int(sha1(self).hexdigest(), 16)
    
    def __eq__(self, other):
        return all(np.ndarray.__eq__(self, other))
    
    def __hash__(self):
        if self.__hash == None:
            self.__hash = int(sha1(self).hexdigest(), 16)
        return self.__hash
    
    def __setitem__(self, key, value):
        raise Exception('hashable arrays are read-only')
        
def quadratic_dot(Q,a,b):
    '''
    Performs to the dot product appropriately
    '''
    return np.tensordot(np.tensordot(Q,a,(1,0)),b,(1,0))
