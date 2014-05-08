# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 17:26:22 2014

@author: dgevans
"""

import numpy as np
from SparseGrid import interpolator as sg_interpolator
from Spline import Spline
from scipy.interpolate import UnivariateSpline
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
            
class interpolator(object):
    '''
    A new interpolator class which removes a lot of the 1 dimensional accuracy
    '''
    def __init__(self,d,mu,Gamma):
        '''
        Initiates the interpolator class
        '''
        #check if there is any variation
        self.f = None
        if np.var(Gamma,0)[0] >0 or d == 1:
            self.f = {}
            Gammahat = Gamma.copy()
            for i in range(1,d):
                self.f[i] = UnivariateSpline(Gamma[:,0],Gamma[:,i])
                Gammahat[:,i] -= self.f[i](Gamma[:,0])
            Mu = np.zeros(d)
            Sigma = np.zeros((d,d))
            for i in range(d):
                xmin,xmax = min(Gammahat[:,i]),max(Gammahat[:,i])
                Mu[i] = (xmin+xmax)/2
                Sigma[i,i] = (xmax-xmin)/2
            self.interp = sg_interpolator(d,mu,Mu,Sigma)
            self.d = d
        else:
            Mu = np.zeros(d)
            Sigma = np.zeros((d,d))
            for i in range(d):
                xmin,xmax = min(Gamma[:,i])-0.001,max(Gamma[:,i])+0.001
                Mu[i] = (xmin+xmax)/2
                Sigma[i,i] = (xmax-xmin)/2
            self.interp = sg_interpolator(d,mu,Mu,Sigma)
            
    def getX(self):
        '''
        Gets the domain where the function needs to be evaluated
        '''
        if self.f == None:
            return self.interp.getX()
        else:
            X = self.interp.getX()
            for i in range(1,self.d):
                X[:,i] += self.f[i](X[:,0])
            return X
            
    def fit(self,Fs):
        '''
        Fits the vector of data X
        '''
        if self.f == None:
            return self.interp.fit(Fs)
        else:
            return IFunction(self.f,self.interp.fit(Fs),self.d)
            
class IFunction(object):
    '''
    Defines a function given the new domain
    '''
    def __init__(self,f,F,d):
        '''
        Initializes
        '''
        self.f = f
        self.F = F
        self.d = d
    def __call__(self,X):
        '''
        Evaluates the function
        '''
        X = np.atleast_2d(X.copy())
        for i in range(1,self.d):
            X[:,i] -= self.f[i](X[:,0])
        return self.F(X)
            
class interpolator_factory(object):
    '''
    Generates an interpolator factory which will interpolate vector functions
    '''
    def __init__(self,d,mu,Gamma):
        '''
        Inits with types, orders and k
        '''
        self.interpolate = interpolator(d,mu,Gamma)
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
    
    
class dict_fun(object):
    '''
    Creates a copy function which stores the results in a dictionary
    '''
    def __init__(self,f):
        '''
        Initialize with function f
        '''
        self.f = f
        self.fm = {}
    def __call__(self,z):
        '''
        Evaluates at a point z, first checks if z in in dictionary, if so, returns
        stored result.        
        '''
        hash_z = sha1(np.ascontiguousarray(z)).hexdigest()
        if self.fm.has_key(hash_z):
            return self.fm[hash_z]
        else:
            f = self.f(z)
            self.fm[hash_z] = f
            return f

