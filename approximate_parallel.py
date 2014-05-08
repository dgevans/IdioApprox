# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 17:28:52 2014

@author: dgevans
"""
from copy import copy
import steadystate
import numpy as np

import utilities
from utilities import hashable_array
from utilities import quadratic_dot
from utilities import dict_fun
import itertools
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def parallel_map(f,X):
    '''
    A map function that applies f to each element of X
    '''
    s = comm.Get_size() #gets the number of processors
    nX = len(X)/s
    r = len(X)%s
    my_range = range(nX*rank+min(rank,r),nX*(rank+1)+min(rank+1,r))
    my_data =  map(f,X[my_range])
    data = comm.allgather(my_data)
    return list(itertools.chain(*data))
    
def parallel_sum(f,X):
    '''
    In parallel applies f to each element of X and computes the sum
    '''
    s = comm.Get_size() #gets the number of processors
    nX = len(X)/s
    r = len(X)%s
    my_range = range(nX*rank+min(rank,r),nX*(rank+1)+min(rank+1,r))
    my_sum =  sum(itertools.imap(f,X[my_range]))
    return sum( comm.allgather(my_sum) )
    
def parallel_dict_map(F,l):
    '''
    perform a map preserving the dict structure
    '''
    ret = {}
    temp = parallel_map(F,l)
    keys = temp[0].keys()
    for key in keys:
        ret[key] = [t[key] for t in temp]
    return ret
    
    
F =None
G = None
f = None

n = None
ny = None
ne = None
nY = None
nz = None
nv = None
Ivy = None
Izy = None

y,e,Y,z,v,eps,S,sigma = [None]*8
#interpolate = utilities.interpolator_factory([3])

def calibrate(Para):
    global F,G,f,ny,ne,nY,nz,nv,n,Ivy,Izy
    global y,e,Y,z,v,eps,S,sigma
    #global interpolate
    F,G,f = Para.F,Para.G,Para.f
    ny,ne,nY,nz,nv,n = Para.ny,Para.ne,Para.nY,Para.nz,Para.nv,Para.n
    Ivy,Izy = np.zeros((nv,ny)),np.zeros((nz,ny)) # Ivy given a vector of y_{t+1} -> v_t and Izy picks out the state
    Ivy[:,-nv:] = np.eye(nv)
    Izy[:,:nz] = np.eye(nz)
        
    
    #store the indexes of the various types of variables
    y = np.arange(ny).view(hashable_array)
    e = np.arange(ny,ny+ne).view(hashable_array)
    Y = np.arange(ny+ne,ny+ne+nY).view(hashable_array)
    z = np.arange(ny+ne+nY,ny+ne+nY+nz).view(hashable_array)
    v = np.arange(ny+ne+nY+nz,ny+ne+nY+nz+nv).view(hashable_array)
    eps = np.arange(ny+ne+nY+nz+nv,ny+ne+nY+nz+nv+1).view(hashable_array)
    
    S = np.hstack((z,Y)).view(hashable_array)
    
    sigma = Para.sigma_e
    
    #interpolate = utilities.interpolator_factory([3]*nz) # cubic interpolation
    steadystate.calibrate(Para)

class approximate(object):
    '''
    Computes the second order approximation 
    '''
    def __init__(self,Gamma):
        '''
        Approximate the equilibrium policies given z_i
        '''
        self.Gamma = Gamma
        self.ss = steadystate.steadystate(Gamma.T)
        
        #x = []
        #for i in range(nz):
        #    x.append(np.linspace(min(Gamma[:,i])-0.05,0.05+max(Gamma[:,i]),20))
        #self.zgrid =  Spline.makeGrid(x)
        #Mu = np.zeros(nz)
        #Sigma = np.zeros((nz,nz))
        #for i in range(nz):
        #    xmin,xmax = min(Gamma[:,i])-0.001,max(Gamma[:,i])+0.001
        #    Mu[i] = (xmin+xmax)/2
        #    Sigma[i,i] = (xmax-xmin)/2
        self.interpolate = utilities.interpolator_factory(nz,3,Gamma)
        self.zgrid = self.interpolate.X
        #precompute Jacobians and Hessians
        self.DF = dict_fun(lambda z_i:utilities.ad_Jacobian(F,self.get_w(z_i)))
        self.HF = dict_fun(lambda z_i:utilities.ad_Hessian(F,self.get_w(z_i)))
        
        self.DG = dict_fun(lambda z_i:utilities.ad_Jacobian(G,self.get_w(z_i)))
        self.HG = dict_fun(lambda z_i:utilities.ad_Hessian(G,self.get_w(z_i)))
        
        self.df = dict_fun(lambda z_i:utilities.ad_Jacobian(f,self.get_w(z_i)[y]))
        self.Hf = dict_fun(lambda z_i:utilities.ad_Hessian(f,self.get_w(z_i)[y]))
        
        #linearize
        self.linearize()
        self.quadratic()
        
    def integrate(self,f):
        '''
        Integrates a function f over Gamma
        '''
        return parallel_sum(f,self.Gamma)/len(self.Gamma)
    
    def get_w(self,z_i):
        '''
        Gets w for particular z_i
        '''
        ybar = self.ss.get_y(z_i).flatten()
        Ybar = self.ss.get_Y()
        ebar = f(ybar)
        
        return np.hstack((
        ybar,ebar,Ybar,ybar[:nz],Ivy.dot(ybar),0.
        ))
        
    def compute_dy(self,z_i):
        '''
        Computes dy w.r.t z_i, eps, Y
        '''
        
        DF = self.DF(z_i)
        df = self.df(z_i)
        
        dy = {}
        DFi = DF[n:] # pick out the relevant equations from F. Forexample to compute dy_{z_i} we need to drop the first equation
        dy[z] = np.linalg.solve(DFi[:,y] + DFi[:,e].dot(df)+DFi[:,v].dot(Ivy),
                                    -DFi[:,z])
        DFi = DF[:-n,:]
        dy[eps] = np.linalg.solve(DFi[:,y] + DFi[:,v].dot(Ivy).dot(dy[z]).dot(Izy),
                                    -DFi[:,eps])
                                    
        DFi = DF[n:,:]
        dy[Y] = np.linalg.solve(DFi[:,y] + DFi[:,e].dot(df) + DFi[:,v].dot(Ivy),
                                -DFi[:,Y])
                                
        return dy
        
    def linearize(self):
        '''
        Computes the linearization
        '''
        self.dy = {}
        temp = parallel_dict_map(self.compute_dy,self.zgrid)
        for x in [z,eps,Y]:
            self.dy[x] = dict_fun(self.interpolate(temp[x]))
        
        DG = self.DG
        
        def DGY_int(z_i):
            DGi = DG(z_i)
            return DGi[:,Y]+DGi[:,y].dot(self.dy[Y](z_i))
        
        self.DGY = self.integrate(DGY_int)
        
        def dYf(z_i):
            DGi = DG(z_i)
            return np.linalg.solve(self.DGY,
                            -DGi[:,z]-DGi[:,y].dot(self.dy[z](z_i)))
        
        self.dY = dict_fun(self.interpolate(
                              parallel_map(dYf,self.zgrid)))
                              
    def get_d(self,z_i):
        '''
        Gets linear constributions
        '''
        dy = self.dy
        d = {}
        df = self.df(z_i)
        
        d[y,S],d[y,eps] = np.hstack((dy[z](z_i),dy[Y](z_i))),dy[eps](z_i) # first order effect of S and eps on y
    
        d[e,S],d[e,eps] = df.dot(d[y,S]), np.zeros((ne,1)) # first order effect of S on e
    
        d[Y,S],d[Y,eps] = np.hstack(( np.zeros((nY,nz)), np.eye(nY) )),np.zeros((nY,1))
        
        d[z,S],d[z,eps] = np.hstack(( np.eye(nz), np.zeros((nz,nY)) )),np.zeros((nz,1))
        
        d[v,S],d[v,eps] = Ivy.dot(d[y,S]), Ivy.dot(dy[z](z_i)).dot(Izy).dot(dy[eps](z_i))
        
        d[eps,S],d[eps,eps] = np.zeros((1,nz+nY)),np.eye(1)       

        d[y,z] = d[y,S][:,:nz]
        
        d[z,eps] = Izy.dot(d[y,eps])

        return d
             
    def compute_d2y(self,z_i):
        '''
        Computes second derivative of y
        '''
        DF,HF = self.DF(z_i),self.HF(z_i)
        df,Hf = self.df(z_i),self.Hf(z_i)
        
        #first compute DFhat, need loadings of S and epsilon on variables
        d = self.get_d(z_i)
        #Now compute Fhat
        HFhat = {}
        HFhat[S,S],HFhat[S,eps],HFhat[eps,S],HFhat[eps,eps] = 0.,0.,0.,0.
        for x1 in [y,e,Y,z,v,eps]:
            HFx1 = HF[:,x1,:]
            for x2 in [y,e,Y,z,v,eps]:
                HFhat[S,S] += quadratic_dot(HFx1[n:,:,x2],d[x1,S],d[x2,S])
                HFhat[S,eps] += quadratic_dot(HFx1[:-n,:,x2],d[x1,S],d[x2,eps])
                HFhat[eps,S] += quadratic_dot(HFx1[:-n,:,x2],d[x1,eps],d[x2,S])
                HFhat[eps,eps] += quadratic_dot(HFx1[:-n,:,x2],d[x1,eps],d[x2,eps])
                
        #Now compute d2y
        d2y = {}
        DFi = DF[n:]
        d2y[S,S] = np.tensordot(np.linalg.inv(DFi[:,y] + DFi[:,e].dot(df) + DFi[:,v].dot(Ivy)),
        -HFhat[S,S] - np.tensordot(DFi[:,e],quadratic_dot(Hf,d[y,S],d[y,S]),1)
        , axes=1)
        
        DFi = DF[:-n]
        d2y[S,eps] =np.tensordot(np.linalg.inv(DFi[:,y] + DFi[:,v].dot(Ivy).dot(d[y,z]).dot(Izy)),
        -HFhat[S,eps] - np.tensordot(DFi[:,v].dot(Ivy), d2y[S,S][:,:,:nz].dot(d[z,eps]),1)
        , axes=1)
        d2y[eps,S] = copy(d2y[S,eps])
        d2y[eps,S].transpose(0,2,1)
        
        d2y[eps,eps] = np.tensordot(np.linalg.inv(DFi[:,y] + DFi[:,v].dot(Ivy).dot(d[y,z]).dot(Izy)),
        -HFhat[eps,eps] - np.tensordot(DFi[:,v].dot(Ivy), 
        quadratic_dot(d2y[S,S][:,:nz,:nz],d[z,eps],d[z,eps]),1)
        ,axes=1)
        
        return d2y
    
    def quadratic(self):
        '''
        Computes the quadratic approximation
        '''
        self.d2y = {}
        temp = parallel_dict_map(self.compute_d2y,self.zgrid)
        for x1 in [S,eps]:
            for x2 in [S,eps]:
                self.d2y[x1,x2] = dict_fun(self.interpolate(temp[x1,x2]))
                
        #Now d2Y
        DGhat_f = self.interpolate(parallel_map(self.compute_DGhat,self.zgrid))
        DGhat = {}
        DGhat[z,z] = dict_fun(DGhat_f[:,:nz,:nz])
        DGhat[z,Y] = dict_fun(DGhat_f[:,:nz,nz:])
        DGhat[Y,z] = dict_fun(DGhat_f[:,nz:,:nz])
        DGhat[Y,Y] = self.integrate(DGhat_f[:,nz:,nz:])
        self.compute_d2Y(DGhat)
        self.compute_dsigma()
        

                
    def compute_d2Y(self,DGhat):
        '''
        Computes components of d2Y
        '''
        d2Y = {}
        DGYinv= np.linalg.inv(self.DGY)
        #First z_i,z_i
        d2Y[z,z] = dict_fun(self.interpolate(parallel_map(lambda z_i:
            np.tensordot(DGYinv, - DGhat[z,z](z_i),1),self.zgrid)))
            
        d2Y[Y,z] = dict_fun(self.interpolate(parallel_map(lambda z_i:
            np.tensordot(DGYinv, - DGhat[Y,z](z_i)
            -DGhat[Y,Y].dot(self.dY(z_i))/2,1),self.zgrid ) ))
            
        d2Y[z,Y] = dict_fun(d2Y[Y,z].f.transpose(0,2,1))
        self.d2Y = d2Y
        
            
    def compute_DGhat(self,z_i):
        '''
        Computes the second order approximation for agent of type z_i
        '''
        DG,HG = self.DG(z_i),self.HG(z_i)
        d = self.get_d(z_i)
        d2y = self.d2y
        
        DGhat = np.zeros((nY,nz+nY,nz+nY))
        DGhat += np.tensordot(DG[:,y],d2y[S,S](z_i),1)
        for x1 in [y,Y,z]:
            HGx1 = HG[:,x1,:]
            for x2 in [y,Y,z]:
                DGhat += quadratic_dot(HGx1[:,:,x2],d[x1,S],d[x2,S])
        return DGhat
        
    def compute_d2y_sigma(self,z_i,integral_term):
        '''
        Computes linear contribution of sigma, dYsigma and dY1sigma
        '''
        DF = self.DF(z_i)
        df,Hf = self.df(z_i),self.Hf(z_i)
        #first compute DFhat, need loadings of S and epsilon on variables
        d = self.get_d(z_i)
        d[y,Y] = d[y,S][:,nz:]
        DFi = DF[n:] #conditions like x_i = Ex_i don't effect this
        
        
        temp = np.linalg.inv(DFi[:,y]+DFi[:,e].dot(df)+DFi[:,v].dot(Ivy+Ivy.dot(d[y,z]).dot(Izy)))
        
        Ahat = (-DFi[:,e].dot(df).dot(self.d2y[eps,eps](z_i).flatten()) 
        -DFi[:,e].dot(quadratic_dot(Hf,d[y,eps],d[y,eps]).flatten()) 
        -DFi[:,v].dot(Ivy).dot(d[y,Y]).dot(integral_term) )
        
        Bhat = -DFi[:,Y]
        Chat = -DFi[:,v].dot(Ivy).dot(d[y,Y])
        return temp.dot(np.hstack((Ahat.reshape(-1,1),Bhat,Chat)))
        
    def compute_DGhat_sigma(self,z_i):
        '''
        Computes the second order approximation for agent of type z_i
        '''
        DG,HG = self.DG(z_i),self.HG(z_i)
        d = self.get_d(z_i)
        d2y = self.d2y
        
        DGhat = np.zeros(nY)
        DGhat += DG[:,y].dot(d2y[eps,eps](z_i).flatten())
        d[eps,eps] =np.array([[1]])
        for x1 in [y,eps]:
            HGx1 = HG[:,x1,:]
            for x2 in [y,eps]:
                DGhat += quadratic_dot(HGx1[:,:,x2],d[x1,eps],d[x2,eps]).flatten()
        return DGhat
        
    def compute_dsigma(self):
        '''
        Computes how dY and dy_i depend on sigma
        '''
        DG = self.DG
        #Now how do things depend with sigma
        integral_term =self.integrate(lambda z_i:
            quadratic_dot(self.d2Y[z,z](z_i),Izy.dot(self.dy[eps](z_i)),Izy.dot(self.dy[eps](z_i))).flatten()
            + self.dY(z_i).dot(Izy).dot(self.d2y[eps,eps](z_i).flatten()))
            
        ABCi = self.interpolate(parallel_map(lambda z_i:self.compute_d2y_sigma(z_i,integral_term),self.zgrid) )
        Ai,Bi,Ci = dict_fun(ABCi[:,0]),dict_fun(ABCi[:,1:nY+1]),dict_fun(ABCi[:,nY+1:])
        Atild = self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(Ai(z_i)))
        Btild = self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(Bi(z_i)))
        Ctild = self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(Ci(z_i)))
        tempC = np.linalg.inv(np.eye(nY)-Ctild)

        DGhat = self.integrate(self.interpolate(parallel_map(self.compute_DGhat_sigma,self.zgrid)))
        
        temp1 = self.integrate(lambda z_i:DG(z_i)[:,Y] + DG(z_i)[:,y].dot(Bi(z_i)+Ci(z_i).dot(tempC).dot(Btild)) )
        temp2 = self.integrate(lambda z_i:DG(z_i)[:,y].dot(Ai(z_i)+Ci(z_i).dot(tempC).dot(Atild)) )
        
        self.d2Y[sigma] = np.linalg.solve(temp1,-DGhat-temp2)
        self.d2y[sigma] = dict_fun(self.interpolate(
                parallel_map(lambda z_i: Ai(z_i) + Ci(z_i).dot(tempC).dot(Atild) +
                      ( Bi(z_i)+Ci(z_i).dot(tempC).dot(Btild) ).dot(self.d2Y[sigma]),self.zgrid)))
        
    def get_approximation(self):
        '''
        Returns approximation object
        '''
        return approximation(self.dy,self.d2y,self.dY,self.d2Y,self.ss)
        
    def iterate(self):
        '''
        Iterates the distribution by randomly sampling
        '''
        def compute_ye(z_i):
            e = np.random.randn()*sigma
            return np.hstack(( self.ss.get_y(z_i).flatten() + self.dy[eps](z_i).flatten()*e + 0.5*(self.d2y[eps,eps](z_i).flatten()*e**2 + self.d2y[sigma](z_i).flatten()*sigma**2).flatten()
                               ,e))
            
        ye = np.vstack(parallel_map(compute_ye,self.Gamma))
        y,epsilon = ye[:,:-1],ye[:,-1]
        Gamma = y.dot(Izy.T)
        Y = self.ss.get_Y() + 0.5*self.d2Y[sigma].flatten()*sigma**2
        
        return Gamma,Y,y,epsilon
        
        
        
class approximation(object):
    '''
    Holds all the relevant infomation of the approximate class
    '''
    def __init__(self,dy,d2y,dY,d2Y,ss):
        '''
        Stores these variables
        '''
        self.dy,self.d2y,self.dY,self.d2Y,self.ss = dy,d2y,dY,d2Y,ss
