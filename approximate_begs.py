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
from scipy.cluster.vq import kmeans2

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
    my_range = slice(nX*rank+min(rank,r),nX*(rank+1)+min(rank+1,r))
    my_data =  map(f,X[my_range])
    data = comm.gather(my_data)
    data = comm.bcast(data)
    return list(itertools.chain(*data))
    
def parallel_sum(f,X):
    '''
    In parallel applies f to each element of X and computes the sum
    '''
    s = comm.Get_size() #gets the number of processors
    nX = len(X)/s
    r = len(X)%s
    my_range = slice(nX*rank+min(rank,r),nX*(rank+1)+min(rank+1,r))
    my_sum =  sum(itertools.imap(f,X[my_range]))
    sums = comm.gather(my_sum)
    return sum( comm.bcast(sums) )
    
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
nG = None
ny = None
ne = None
nY = None
nz = None
nv = None
n_p = None
neps = None
Ivy = None
Izy = None
Para = None

shock = None

logm_min = -6.
muhat_min = -np.inf

y,e,Y,z,v,eps,Eps,p,S,sigma,sigma_E = [None]*11
#interpolate = utilities.interpolator_factory([3])

def calibrate(Parahat):
    global F,G,f,ny,ne,nY,nz,nv,n,Ivy,Izy,nG,n_p,neps
    global y,e,Y,z,v,eps,p,S,sigma,Eps,sigma_E,Para
    Para = Parahat
    #global interpolate
    F,G,f = Para.F,Para.G,Para.f
    ny,ne,nY,nz,nv,n,nG,n_p,neps = Para.ny,Para.ne,Para.nY,Para.nz,Para.nv,Para.n,Para.nG,Para.n_p,Para.neps
    Ivy,Izy = np.zeros((nv,ny)),np.zeros((nz,ny)) # Ivy given a vector of y_{t+1} -> v_t and Izy picks out the state
    Ivy[:,-nv:] = np.eye(nv)
    Izy[:,:nz] = np.eye(nz)
        
    
    #store the indexes of the various types of variables
    y = np.arange(ny).view(hashable_array)
    e = np.arange(ny,ny+ne).view(hashable_array)
    Y = np.arange(ny+ne,ny+ne+nY).view(hashable_array)
    z = np.arange(ny+ne+nY,ny+ne+nY+nz).view(hashable_array)
    v = np.arange(ny+ne+nY+nz,ny+ne+nY+nz+nv).view(hashable_array)
    p = np.arange(ny+ne+nY+nz+nv,ny+ne+nY+nz+nv+n_p).view(hashable_array)
    eps = np.arange(ny+ne+nY+nz+nv+n_p,ny+ne+nY+nz+nv+n_p+neps).view(hashable_array)
    Eps = np.arange(ny+ne+nY+nz+nv+n_p+neps,ny+ne+nY+nz+nv+n_p+neps+1).view(hashable_array)
    
    S = np.hstack((z,Y)).view(hashable_array)
    
    sigma = Para.sigma_e.view(hashable_array)
    
    sigma_E = Para.sigma_E
    
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
        self.approximate_Gamma()
        self.ss = steadystate.steadystate(self.Gamma_ss.T)

        #precompute Jacobians and Hessians
        self.get_w = dict_fun(self.get_wf)
        self.DF = dict_fun(lambda z_i:utilities.ad_Jacobian(F,self.get_w(z_i)))
        self.HF = dict_fun(lambda z_i:utilities.ad_Hessian(F,self.get_w(z_i)))
        
        self.DG = dict_fun(lambda z_i:utilities.ad_Jacobian(G,self.get_w(z_i)))
        self.HG = dict_fun(lambda z_i:utilities.ad_Hessian(G,self.get_w(z_i)))
        
        self.df = dict_fun(lambda z_i:utilities.ad_Jacobian(f,self.get_w(z_i)[y]))
        self.Hf = dict_fun(lambda z_i:utilities.ad_Hessian(f,self.get_w(z_i)[y]))
        
        
        #linearize
        self.linearize()
        self.quadratic()
        self.join_function()
        
    def approximate_Gamma(self,k=400):
        '''
        Approximate the Gamma distribution
        '''
        if rank == 0:
            cluster,labels = kmeans2(self.Gamma,k,minit='points')
            cluster,labels = comm.bcast((cluster,labels))
        else:
            cluster,labels = comm.bcast(None)
        weights = (labels-np.arange(k).reshape(-1,1) ==0).sum(1)/float(len(self.Gamma))
        #Para.nomalize(cluster,weights)
        self.Gamma_ss = cluster[labels,:]
        mask = weights > 0
        cluster = cluster[mask]
        weights = weights[mask]
        
        
        self.dist = zip(cluster,weights)        
        
        
    def approximate_Gamma_old(self,crit = 0.4):
        '''
        Approximate the Gamma distribution
        '''
        Gamma = self.Gamma
        points = []
        mapped_points = []
        test = map(lambda z: set(np.arange(len(Gamma))[np.linalg.norm(Gamma-z,np.inf,1)<crit]),Gamma)
        while True:
            i = np.argmax(map(len,test))
            if len(test[i]) == 0:
                break
            points.append(i)
            mapped_points.append(list(test[i]))
            test = map(lambda t: t.difference(mapped_points[-1]),test)

        points = np.hstack(points)
        
        self.Gamma_ss = np.empty(self.Gamma.shape)
        for p,mp in itertools.izip(points,mapped_points):
            self.Gamma_ss[mp,:] = self.Gamma[p]
            
        Gammabar = self.Gamma[points,:]
        weights = np.hstack(map(len,mapped_points))/float(len(self.Gamma))
        self.dist = zip(Gammabar,weights)

        
    def integrate(self,f):
        '''
        Integrates a function f over Gamma
        '''
        def f_int(x):
            z,w = x
            return w*f(z)
        return parallel_sum(f_int,self.dist)
    
    def get_wf(self,z_i):
        '''
        Gets w for particular z_i
        '''
        ybar = self.ss.get_y(z_i).flatten()
        Ybar = self.ss.get_Y()
        ebar = f(ybar)
        
        return np.hstack((
        ybar,ebar,Ybar,ybar[:nz],Ivy.dot(ybar),1.,np.zeros(neps),0.
        ))
        
    def compute_dy(self):
        '''
        Computes dy w.r.t z_i, eps, Y
        '''
        self.dy = {}
        
        def dy_z(z_i):
            DF = self.DF(z_i)
            df = self.df(z_i)
            DFi = DF[n:] # pick out the relevant equations from F. Forexample to compute dy_{z_i} we need to drop the first equation
            return np.linalg.solve(DFi[:,y] + DFi[:,e].dot(df)+DFi[:,v].dot(Ivy),
                                    -DFi[:,z])
        self.dy[z] = dict_fun(dy_z)
        
        def dy_eps(z_i):   
            DF = self.DF(z_i)    
            DFi = DF[:-n,:]
            return np.linalg.solve(DFi[:,y] + DFi[:,v].dot(Ivy).dot(self.dy[z](z_i)).dot(Izy),
                                    -DFi[:,eps])
                                    
        self.dy[eps] = dict_fun(dy_eps)
        
        def dy_Eps(z_i):   
            DF = self.DF(z_i)    
            DFi = DF[:-n,:]
            return np.linalg.solve(DFi[:,y] + DFi[:,v].dot(Ivy).dot(self.dy[z](z_i)).dot(Izy),
                                    -DFi[:,Eps])
                                    
        self.dy[Eps] = dict_fun(dy_Eps)
                
        def dy_Y(z_i):
            DF = self.DF(z_i)
            df = self.df(z_i)                    
            DFi = DF[n:,:]
            return np.linalg.solve(DFi[:,y] + DFi[:,e].dot(df) + DFi[:,v].dot(Ivy),
                                -DFi[:,Y])
        self.dy[Y] = dict_fun(dy_Y)
        
        def dy_YEps(z_i):   
            DF = self.DF(z_i)    
            DFi = DF[:-n,:]
            return np.linalg.solve(DFi[:,y] + DFi[:,v].dot(Ivy).dot(self.dy[z](z_i)).dot(Izy),
                                    -DFi[:,Y])
                                    
        self.dy[Y,Eps] = dict_fun(dy_YEps)
        
    def linearize(self):
        '''
        Computes the linearization
        '''
        self.compute_dy()
        
        DG = lambda z_i : self.DG(z_i)[nG:,:] #account for measurability constraints
        
        def DGY_int(z_i):
            DGi = DG(z_i)
            return DGi[:,Y]+DGi[:,y].dot(self.dy[Y](z_i))
        
        self.DGYinv = np.linalg.inv(self.integrate(DGY_int))
        
        def dYf(z_i):
            DGi = DG(z_i)
            return self.DGYinv.dot(-DGi[:,z]-DGi[:,y].dot(self.dy[z](z_i)))
        
        self.dY = dict_fun(dYf)
        
        self.linearize_aggregate()
        
        self.linearize_parameter()
    
    def linearize_aggregate(self):
        '''
        Linearize with respect to aggregate shock.
        '''
        dy = {}
        
        def Fhat_y(z_i):
            DFi = self.DF(z_i)[:-n,:]
            return DFi[:,y] + DFi[:,v].dot(Ivy).dot(self.dy[z](z_i)).dot(Izy)
            
        Fhat_inv = dict_fun(lambda z_i: np.linalg.inv(Fhat_y(z_i))) 
        
        def dy_dYprime(z_i):
            DFi = self.DF(z_i)[:-n,:]
            return Fhat_inv(z_i).dot(DFi[:,v]).dot(Ivy).dot(self.dy[Y](z_i))
            
        dy_dYprime = dict_fun(dy_dYprime)
        
        temp = -np.linalg.inv( np.eye(nY) + self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(dy_dYprime(z_i))) )
        
        dYprime = {}
        dYprime[Eps] = temp.dot(
        self.integrate(lambda z_i : self.dY(z_i).dot(Izy).dot(Fhat_inv(z_i)).dot(self.DF(z_i)[:-n,Eps]))        
        )
        dYprime[Y] = temp.dot(
        self.integrate(lambda z_i : self.dY(z_i).dot(Izy).dot(Fhat_inv(z_i)).dot(self.DF(z_i)[:-n,Y]))        
        )
        
        
        dy[Eps] = dict_fun(
        lambda z_i : -Fhat_inv(z_i).dot(self.DF(z_i)[:-n,Eps]) - dy_dYprime(z_i).dot(dYprime[Eps])        
        )
        
        dy[Y] = dict_fun(
        lambda z_i : -Fhat_inv(z_i).dot(self.DF(z_i)[:-n,Y]) - dy_dYprime(z_i).dot(dYprime[Y])         
        )
        
        #Now do derivatives w.r.t G
        DGi = dict_fun(lambda z_i : self.DG(z_i)[:-nG,:])
        
        DGhatinv = np.linalg.inv(self.integrate(
        lambda z_i : DGi(z_i)[:,Y] + DGi(z_i)[:,y].dot(dy[Y](z_i))        
        ))
        
        self.dY_Eps = -DGhatinv.dot( self.integrate(
        lambda z_i : DGi(z_i)[:,Eps] + DGi(z_i)[:,y].dot(dy[Eps](z_i))          
        ) )
        
        self.dy[Eps] = dict_fun(
        lambda z_i : dy[Eps](z_i) + dy[Y](z_i).dot(self.dY_Eps)
        )
        
    def linearize_parameter(self):
        '''
        Linearize with respect to aggregate shock.
        '''
        dy = {}
        
        def Fhat_p(z_i):
            DFi = self.DF(z_i)[n:,:]
            df = self.df(z_i)
            return DFi[:,y]+ DFi[:,e].dot(df) + DFi[:,v].dot(Ivy) + DFi[:,v].dot(Ivy).dot(self.dy[z](z_i)).dot(Izy)
            
        Fhat_inv = dict_fun(lambda z_i: np.linalg.inv(Fhat_p(z_i))) 
        
        def dy_dYprime(z_i):
            DFi = self.DF(z_i)[n:,:]
            return Fhat_inv(z_i).dot(DFi[:,v]).dot(Ivy).dot(self.dy[Y](z_i))
            
        dy_dYprime = dict_fun(dy_dYprime)
        
        temp = -np.linalg.inv( np.eye(nY) + self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(dy_dYprime(z_i))) )
        
        dYprime = {}
        dYprime[p] = temp.dot(
        self.integrate(lambda z_i : self.dY(z_i).dot(Izy).dot(Fhat_inv(z_i)).dot(self.DF(z_i)[n:,p]))        
        )
        dYprime[Y] = temp.dot(
        self.integrate(lambda z_i : self.dY(z_i).dot(Izy).dot(Fhat_inv(z_i)).dot(self.DF(z_i)[n:,Y]))        
        )
        
        
        dy[p] = dict_fun(
        lambda z_i : -Fhat_inv(z_i).dot(self.DF(z_i)[n:,p]) - dy_dYprime(z_i).dot(dYprime[p])        
        )
        
        dy[Y] = dict_fun(
        lambda z_i : -Fhat_inv(z_i).dot(self.DF(z_i)[n:,Y]) - dy_dYprime(z_i).dot(dYprime[Y])         
        )
        
        #Now do derivatives w.r.t G
        DGi = dict_fun(lambda z_i : self.DG(z_i)[nG:,:])
        
        DGhatinv = np.linalg.inv(self.integrate(
        lambda z_i : DGi(z_i)[:,Y] + DGi(z_i)[:,y].dot(dy[Y](z_i))        
        ))
        
        self.dY_p = -DGhatinv.dot( self.integrate(
        lambda z_i : DGi(z_i)[:,p] + DGi(z_i)[:,y].dot(dy[p](z_i))          
        ) )
        
        self.dy[p] = dict_fun(
        lambda z_i : dy[p](z_i) + dy[Y](z_i).dot(self.dY_p)
        )
        
                              
    def get_df(self,z_i):
        '''
        Gets linear constributions
        '''
        dy = self.dy
        d = {}
        df = self.df(z_i)
        
        d[y,S],d[y,eps] = np.hstack((dy[z](z_i),dy[Y](z_i))),dy[eps](z_i) # first order effect of S and eps on y
    
        d[e,S],d[e,eps] = df.dot(d[y,S]), np.zeros((ne,1)) # first order effect of S on e
    
        d[Y,S],d[Y,eps] = np.hstack(( np.zeros((nY,nz)), np.eye(nY) )),np.zeros((nY,neps))
        
        d[z,S],d[z,eps] = np.hstack(( np.eye(nz), np.zeros((nz,nY)) )),np.zeros((nz,neps))
        
        d[v,S],d[v,eps] = Ivy.dot(d[y,S]), Ivy.dot(dy[z](z_i)).dot(Izy).dot(dy[eps](z_i))
        
        d[eps,S],d[eps,eps] = np.zeros((neps,nz+nY)),np.eye(neps)       

        d[y,z] = d[y,S][:,:nz]
        

        return d
     
    def compute_HFhat(self):
        '''
        Constructs the HFhat functions
        '''
        self.HFhat = {}
        
        for x1 in [S,eps]:
            for x2 in [S,eps]:
                
                #Construct a function for each pair x1,x2
                def HFhat_temp(z_i,x1=x1,x2=x2):
                    HF = self.HF(z_i)
                    d = self.get_d(z_i)
                    HFhat = 0.
                    for y1 in [y,e,Y,z,v,eps]:
                        HFy1 = HF[:,y1,:]
                        for y2 in [y,e,Y,z,v,eps]:
                            if x1.__hash__() == S.__hash__() and x2.__hash__() == S.__hash__():
                                HFhat += quadratic_dot(HFy1[n:,:,y2],d[y1,x1],d[y2,x2])
                            else:
                                HFhat += quadratic_dot(HFy1[:-n,:,y2],d[y1,x1],d[y2,x2])
                    return HFhat
                    
                self.HFhat[x1,x2] = dict_fun(HFhat_temp)
             
    def compute_d2y(self):
        '''
        Computes second derivative of y
        '''
        self.d2y = {}
        #DF,HF = self.DF(z_i),self.HF(z_i)
        #df,Hf = self.df(z_i),self.Hf(z_i)
        
        #first compute DFhat, need loadings of S and epsilon on variables
                
        #Now compute d2y
        
        def d2y_SS(z_i):
            DF,df,Hf = self.DF(z_i),self.df(z_i),self.Hf(z_i)
            d = self.get_d(z_i)
            DFi = DF[n:]
            return np.tensordot(np.linalg.inv(DFi[:,y] + DFi[:,e].dot(df) + DFi[:,v].dot(Ivy)),
                -self.HFhat[S,S](z_i) - np.tensordot(DFi[:,e],quadratic_dot(Hf,d[y,S],d[y,S]),1)
                    , axes=1)
        self.d2y[S,S] = dict_fun(d2y_SS)
        
        def d2y_Seps(z_i):
            DF = self.DF(z_i)
            d = self.get_d(z_i)
            DFi = DF[:-n]
            dz_eps= Izy.dot(d[y,eps])
            return np.tensordot(np.linalg.inv(DFi[:,y] + DFi[:,v].dot(Ivy).dot(d[y,z]).dot(Izy)),
            -self.HFhat[S,eps](z_i) - np.tensordot(DFi[:,v].dot(Ivy), self.d2y[S,S](z_i)[:,:,:nz].dot(dz_eps),1)
            , axes=1)
            
        self.d2y[S,eps] = dict_fun(d2y_Seps)
        self.d2y[eps,S] = dict_fun(lambda z_i : self.d2y[S,eps](z_i).transpose(0,2,1))
        
        def d2y_epseps(z_i):
            DF = self.DF(z_i)
            d = self.get_d(z_i)
            DFi = DF[:-n]
            dz_eps= Izy.dot(d[y,eps])
            return np.tensordot(np.linalg.inv(DFi[:,y] + DFi[:,v].dot(Ivy).dot(d[y,z]).dot(Izy)),
            -self.HFhat[eps,eps](z_i) - np.tensordot(DFi[:,v].dot(Ivy), 
            quadratic_dot(self.d2y[S,S](z_i)[:,:nz,:nz],dz_eps,dz_eps),1)
            ,axes=1)
            
        self.d2y[eps,eps] = dict_fun(d2y_epseps)
        
    def quadratic(self):
        '''
        Computes the quadratic approximation
        '''
        self.get_d = dict_fun(self.get_df)
        self.compute_HFhat()
        self.compute_d2y()        
        #Now d2Y
        DGhat_f = dict_fun(self.compute_DGhat)
        self.DGhat = {}
        self.DGhat[z,z] = lambda z_i : DGhat_f(z_i)[:,:nz,:nz]
        self.DGhat[z,Y] = lambda z_i : DGhat_f(z_i)[:,:nz,nz:]
        self.DGhat[Y,z] = lambda z_i : DGhat_f(z_i)[:,nz:,:nz]
        self.DGhat[Y,Y] = self.integrate(lambda z_i : DGhat_f(z_i)[:,nz:,nz:])
        self.compute_d2Y()
        self.compute_dsigma()
        

                
    def compute_d2Y(self):
        '''
        Computes components of d2Y
        '''
        DGhat = self.DGhat
        self.d2Y = {}
        #First z_i,z_i
        self.d2Y[z,z] = dict_fun(lambda z_i: np.tensordot(self.DGYinv, - DGhat[z,z](z_i),1))
            
        self.d2Y[Y,z] = dict_fun(lambda z_i: np.tensordot(self.DGYinv, - DGhat[Y,z](z_i) 
                            -DGhat[Y,Y].dot(self.dY(z_i))/2. ,1) )
            
        self.d2Y[z,Y] = lambda z_i : self.d2Y[Y,z](z_i).transpose(0,2,1)
        
            
    def compute_DGhat(self,z_i):
        '''
        Computes the second order approximation for agent of type z_i
        '''
        DG,HG = self.DG(z_i)[nG:,:],self.HG(z_i)[nG:,:,:]
        d = self.get_d(z_i)
        d2y = self.d2y
        
        DGhat = np.zeros((nY,nz+nY,nz+nY))
        DGhat += np.tensordot(DG[:,y],d2y[S,S](z_i),1)
        for x1 in [y,Y,z]:
            HGx1 = HG[:,x1,:]
            for x2 in [y,Y,z]:
                DGhat += quadratic_dot(HGx1[:,:,x2],d[x1,S],d[x2,S])
        return DGhat
        
    def compute_d2y_sigma(self,z_i):
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
        
        Ahat = (-DFi[:,e].dot(df).dot(self.d2y[eps,eps](z_i).diagonal(0,1,2)) 
        -DFi[:,e].dot(quadratic_dot(Hf,d[y,eps],d[y,eps]).diagonal(0,1,2)) 
        -DFi[:,v].dot(Ivy).dot(d[y,Y]).dot(self.integral_term) )
        
        Bhat = -DFi[:,Y]
        Chat = -DFi[:,v].dot(Ivy).dot(d[y,Y])
        return temp.dot(np.hstack((Ahat.reshape(-1,neps),Bhat,Chat)))
        
    def compute_DGhat_sigma(self,z_i):
        '''
        Computes the second order approximation for agent of type z_i
        '''
        DG,HG = self.DG(z_i)[nG:,:],self.HG(z_i)[nG:,:,:]
        d = self.get_d(z_i)
        d2y = self.d2y
        
        DGhat = np.zeros((nY,neps))
        DGhat += DG[:,y].dot(d2y[eps,eps](z_i).diagonal(0,1,2))
        d[eps,eps] =np.eye(neps)
        for x1 in [y,eps]:
            HGx1 = HG[:,x1,:]
            for x2 in [y,eps]:
                DGhat += quadratic_dot(HGx1[:,:,x2],d[x1,eps],d[x2,eps]).diagonal(0,1,2)
        return DGhat
        
    def compute_dsigma(self):
        '''
        Computes how dY and dy_i depend on sigma
        '''
        DG = lambda z_i : self.DG(z_i)[nG:,:]
        #Now how do things depend with sigma
        self.integral_term =self.integrate(lambda z_i:
            quadratic_dot(self.d2Y[z,z](z_i),Izy.dot(self.dy[eps](z_i)),Izy.dot(self.dy[eps](z_i))).diagonal(0,1,2)
            + self.dY(z_i).dot(Izy).dot(self.d2y[eps,eps](z_i).diagonal(0,1,2)))
            
        ABCi = dict_fun(self.compute_d2y_sigma )
        Ai,Bi,Ci = lambda z_i : ABCi(z_i)[:,:neps], lambda z_i : ABCi(z_i)[:,neps:nY+neps], lambda z_i : ABCi(z_i)[:,nY+neps:]
        Atild = self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(Ai(z_i)))
        Btild = self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(Bi(z_i)))
        Ctild = self.integrate(lambda z_i: self.dY(z_i).dot(Izy).dot(Ci(z_i)))
        tempC = np.linalg.inv(np.eye(nY)-Ctild)

        DGhat = self.integrate(self.compute_DGhat_sigma)
        
        temp1 = self.integrate(lambda z_i:DG(z_i)[:,Y] + DG(z_i)[:,y].dot(Bi(z_i)+Ci(z_i).dot(tempC).dot(Btild)) )
        temp2 = self.integrate(lambda z_i:DG(z_i)[:,y].dot(Ai(z_i)+Ci(z_i).dot(tempC).dot(Atild)) )
        
        self.d2Y[sigma] = np.linalg.solve(temp1,-DGhat-temp2)
        self.d2y[sigma] = dict_fun(lambda z_i: Ai(z_i) + Ci(z_i).dot(tempC).dot(Atild) +
                      ( Bi(z_i)+Ci(z_i).dot(tempC).dot(Btild) ).dot(self.d2Y[sigma]))
                      
    def join_function(self):
        '''
        Joins the data for the dict_maps across functions
        '''
        for f in self.dy.values():
            if hasattr(f, 'join'):
                parallel_map(lambda x: f(x[0]),self.dist)
                f.join()
        for f in self.d2y.values():
            if hasattr(f,'join'):       
                parallel_map(lambda x: f(x[0]),self.dist)
                f.join()
          
        self.dY.join()
        
        for f in self.d2Y.values():
            if hasattr(f,'join'):   
                parallel_map(lambda x: f(x[0]),self.dist)
                f.join()
        
        
    def iterate(self,quadratic = True):
        '''
        Iterates the distribution by randomly sampling
        '''
        if rank == 0:
            r = np.random.randn()
            if not shock == None:
                r = shock
            r = min(3.,max(-3.,r))
            E = r*sigma_E
        else:
            E = None
            
        E = comm.bcast(E)
        phat = Para.phat
        Gamma_dist = zip(self.Gamma,self.Gamma_ss)
        
        Y1hat = parallel_sum(lambda z : self.dY(z[1]).dot((z[0]-z[1]))
                          ,Gamma_dist)/len(self.Gamma)
        
        def Y2_int(x):
            z_i,zbar = x
            zhat = z_i-zbar
            return (quadratic_dot(self.d2Y[z,z](zbar),zhat,zhat) + 2* quadratic_dot(self.d2Y[z,Y](zbar),zhat,Y1hat)).flatten()
        
        Y2hat = parallel_sum(Y2_int,Gamma_dist)/len(self.Gamma)
        
        def compute_ye(x):
            z_i,zbar = x
            zhat = z_i-zbar
            r = np.random.randn(neps)
            for i in range(neps):
                if z_i[0] > logm_min and z_i[1] > muhat_min:
                    r[i] = min(3.,max(-3.,r[i]))
                else:
                    r[i] = min(1.,max(logm_min-z_i[0]+0.1,(muhat_min-z_i[1])/10. +0.1))
            e = r*sigma
            Shat = np.hstack([zhat,Y1hat])
            return np.hstack(( self.ss.get_y(zbar).flatten() + self.dy[eps](zbar).dot(e).flatten() + self.dy[Eps](zbar).flatten()*E
                                + self.dy[p](zbar).dot(phat).flatten()
                                + self.dy[z](zbar).dot(zhat).flatten()
                                + self.dy[Y](zbar).dot(Y1hat).flatten()
                                + 0.5*quadratic*(quadratic_dot(self.d2y[eps,eps](zbar),e,e).flatten() + self.d2y[sigma](zbar).dot(sigma**2).flatten()
                                +quadratic_dot(self.d2y[S,S](zbar),Shat,Shat) + 2*quadratic_dot(self.d2y[S,eps](zbar),Shat,e)
                                +self.dy[Y](zbar).dot(Y2hat)                                
                                ).flatten()
                               ,e))
            
        ye = np.vstack(parallel_map(compute_ye,Gamma_dist))
        y,epsilon = ye[:,:-neps],ye[:,-neps]
        Gamma = y.dot(Izy.T)
        Ynew = (self.ss.get_Y() + Y1hat + self.dY_Eps.flatten() * E + self.dY_p.dot(phat).flatten()
                + 0.5*quadratic*(self.d2Y[sigma].dot(sigma**2) + Y2hat).flatten() )
        
        return Gamma,Ynew,epsilon,y
