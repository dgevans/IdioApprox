# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 17:30:59 2014

@author: dgevans
"""
import numpy as np
from scipy.optimize import root

Finv = None
GSS = None


ny = None
ne = None
nY = None
nz = None
nv = None


def calibrate(Para):
    global Finv,GSS,ny,ne,nY,nz,nv
    Finv,GSS = Para.Finv,Para.GSS
    ny,ne,nY,nz,nv = Para.ny,Para.ne,Para.nY,Para.nz,Para.nv
    
class steadystate(object):
    '''
    Computes the steady state
    '''
    def __init__(self,z_i):
        '''
        Solves for the steady state given a distribution z_i
        '''
        self.z_i = z_i
        self.solveSteadyState()
        
    def solveSteadyState(self):
        '''
        Solve for the steady state
        '''
        res = root(self.SteadyStateRes,np.ones(nY))
        if not res.success:
            raise Exception('Could not find steady state')
        self.Y = res.Y
        
    def SteadyStateRes(self,Y):
        '''
        For a given vector of aggregates returns the steady state residual
        '''
        y_i = Finv(Y,self.z_i)
        return GSS(Y,y_i,self.z_i)
        
    def get_Y(self):
        '''
        Gets the aggregate variables for the steady state
        '''
        return self.Y
        
    def get_y(self,z):
        '''
        Given idiosyncratic state z returns the idiosyncratic steady state values
        y
        '''
        return Finv(self.Y,z)