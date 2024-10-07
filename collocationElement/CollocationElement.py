#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   FEM/element.py
@Time    :   2024/02/09 11:25:21
@Author  :   Harley Hanes 
@Version :   1.0
@Contact :   hhanes@ncsu.edu
@License :   (C)Copyright 2023, Harley Hanes
@Desc    :   None
'''
import numpy as np

class Element:
    #Define Attributes
    _nCollocation=""
    _bounds=""
    _spacing=""
    _collocationPoints=""
    _interpolationPoints=""
    _basisCoeff=""
    
    
    def __init__(self,nCollocation=1,bounds =[0,1],spacing="uniform"):
        self.nCollocation=nCollocation
        self.bounds=bounds
        self.spacing=spacing
        
        self.__setCollocationPoints__()
        self.__solveBasisCoeff__()
        
        
        
    @property
    def order(self):
        return self._order
    @order.setter
    def order(self,value):
        if not isinstance(value,int):
            raise Exception("Non-int value entered for order: " + str(value))
        elif value <= 0:
            raise Exception("Non-positive integer entered for order: " +value)
        self._order=value
        
    @property
    def spacing(self):
        return self._spacing
    @spacing.setter
    def spacing(self,value):
        if not isinstance(value,str):
            raise Exception("Non-string value entered for spacing: " + str(value))
        elif value.lower() != "uniform" and value.lower() !="legendre":
            raise Exception("Non uniform or legendre value for spacing: " + str(value))
        self._spacing=value
    
    @property
    def bounds(self):
        return self._bounds
    @bounds.setter
    def bounds(self,value):
        if type(value)!=np.ndarray and type(value)!=list:
            raise Exception("Non-list or numpy array entered for bounds: " + str(value))
        elif type(value)==np.ndarray:
            if value.size!=2 or value.ndim!=1:
                raise Exception("Numpy array of non 1D with size 2 entered for bounds: " + str(value))
            else:
                self._bounds=value
        elif type(value)==list:
            if len(value)!=2:
                raise Exception("List of length other than 2 entered for bounds: " + str(value))
            else:
                self._bounds=value
    @property
    def collocationPoints(self):
        return self._collocationPoints
    @collocationPoints.setter
    def collocationPoints(self,value):
        if type(value)!=np.ndarray and type(value)!=list:
            raise Exception("Non-list or numpy array entered for collocationPoints: " + str(value))
        elif type(value)==np.ndarray:
            if value.size!=self.nCollocation or value.ndim!=1:
                raise Exception("Numpy array of non 1D with size order entered for collocationPoints: " + str(value))
            else:
                self._collocationPoints=value
        elif type(value)==list:
            if len(value)!=self.nCollocation:
                raise Exception("List of length other than order entered for collocationPoints: " + str(value))
            else:
                self._collocationPoints=value
    
    @property
    def interpolationPoints(self):
        return self._interpolationPoints
    @interpolationPoints.setter
    def interpolationPoints(self,value):
        if type(value)!=np.ndarray and type(value)!=list:
            raise Exception("Non-list or numpy array entered for interpolationPoints: " + str(value))
        elif type(value)==np.ndarray:
            if value.size!=self.nCollocation+2 or value.ndim!=1:
                raise Exception("Numpy array of non 1D with size order+2 entered for interpolationPoints: " + str(value))
            else:
                self._interpolationPoints=value
        elif type(value)==list:
            if len(value)!=self.nCollocation+2:
                raise Exception("List of length other than order+2 entered for interpolationPoints: " + str(value))
            else:
                self._interpolationPoints=value
                
    
    @property
    def basisCoeff(self):
        return self._basisCoeff
    @basisCoeff.setter
    def basisCoeff(self,value):
        if type(value)!=np.ndarray:
            raise Exception("Non-numpy array entered for basisCoeff: " + str(value))
        elif value.shape[0]!=self.nCollocation+2 or value.shape[1]!=self.nCollocation+2:
            raise Exception("Numpy array of non order+2 by order+2 entered for basisCoeff: " + str(value))
        else:
            self._basisCoeff=value

                
    
    #Write mappings between [-1,1] and element boundaries
    
    def mapToElementBounds(self,points):
        return self.bounds[0]+(points+1)*(self.bounds[1]-self.bounds[0])/2
    
    def mapFromElementBounds(self,points):
        return 1+2*(points-self.bounds[0])/(self.bounds[1]-self.bounds[0])
    
    #Setup Collocation Points
    def __setCollocationPoints__(self):
        if self.spacing=="uniform":
            self.collocationPoints=np.linspace(self.bounds[0],self.bounds[1],num=self.nCollocation+2)[1:self.nCollocation+1]
        elif self.spacing=="legendre":
            self.collocationPoints=self.mapToElementBounds(np.polynomial.legendre.leggauss(self.nCollocation)[0])
        interpolationPoints = np.empty((self.nCollocation+2))
        interpolationPoints[0]=self.bounds[0]
        interpolationPoints[1:-1]=self.collocationPoints
        interpolationPoints[-1]=self.bounds[1]
        self.interpolationPoints=interpolationPoints
            
    #Compute the coeffecients for the basis so that they interpolate 1 at their collocation point and 0 at all others
    def __solveBasisCoeff__(self):
        basisCoeff=np.empty((self.nCollocation+2,self.nCollocation+2))
        massMatrix=np.empty((self.nCollocation+2,self.nCollocation+2))
        for iOrder in np.arange(self.nCollocation+2):
            massMatrix[:,iOrder]=self.interpolationPoints**iOrder
 
        for iBasis in np.arange(self.nCollocation+2):
            b=np.zeros((self.nCollocation+2))
            b[iBasis]=1
            basisCoeff[iBasis]=np.linalg.solve(massMatrix,b)
        self.basisCoeff=basisCoeff
        
    def basisFunctions(self,x):
        if type(x)!=np.ndarray:
            basisFunctions=np.empty((self.nCollocation+2,1))
            if type(x)==list:
                x=np.array(x)
            elif type(x)==float or type(x)==int or type(x)==np.float64 or type(x)==np.int64  :
                x=np.array([x])
            else:
                raise Exception("Invalid type entered x: "+str(type(x)))
        elif x.ndim!=1:
                raise Exception("Multi-dimensional array entered for x: " + str(x))
        else :
            #Initialize values for every interpolating polynomial at each point in x
            basisFunctions=np.empty((self.nCollocation+2,x.size))
        
        #Setup matrix of every x value to every power up to the order of the interpolating polynomial
        xExpanded = np.ones((self.nCollocation+2,)+x.shape)
        for iOrder in np.arange(self.nCollocation+2):
            xExpanded[iOrder]=x**iOrder
            
        #Set the value of each interpolating basis to be the 
        for iBasis in np.arange(self.nCollocation+2):
            basisFunctions[iBasis] = np.sum(xExpanded.transpose()*self.basisCoeff[iBasis],axis=-1)
                    
        return basisFunctions
    
    def basisFirstDeriv(self,x):
        if type(x)!=np.ndarray:
            basisFirstDeriv=np.empty((self.nCollocation+2,))
            x = np.array([x])
        elif x.ndim!=1:
                raise Exception("Multi-dimensional array entered for x: " + str(x))
        else :
            basisFirstDeriv=np.empty((self.nCollocation+2,x.size))
        
        xExpanded = np.ones((self.nCollocation+1,)+x.shape)
        for iOrder in np.arange(self.nCollocation+1):
            xExpanded[iOrder]=(iOrder+1)*(x**iOrder)
            
        for iBasis in np.arange(self.nCollocation+2):
            basisFirstDeriv[iBasis] = np.sum(xExpanded.transpose()*self.basisCoeff[iBasis][1:],axis=-1)
                    
        return basisFirstDeriv
    
    def basisSecondDeriv(self,x):
        if type(x)!=np.ndarray:
            basisSecondDeriv=np.empty((self.nCollocation+2,1))
            x = np.array([x])
        elif x.ndim!=1:
                raise Exception("Multi-dimensional array entered for x: " + str(x))
        else :
            basisSecondDeriv=np.empty((self.nCollocation+2,x.size))
        
        xExpanded = np.ones((self.nCollocation,)+x.shape)
        for iOrder in np.arange(self.nCollocation):
            xExpanded[iOrder]=(iOrder+2)*(iOrder+1)*(x**iOrder)
            
        for iBasis in np.arange(self.nCollocation+2):
            basisSecondDeriv[iBasis] = np.sum(xExpanded.transpose()*self.basisCoeff[iBasis][2:],axis=-1)
        
        return basisSecondDeriv
    


    #===============================Utility Functions===================================================
    def integrate(self,f, order="auto"):
        #element.integrate: Approximates integral of callable f using quadrature rule of collocation points
        #  Inputs:
        
        #  Outputs:
        #print("Order: ", order)
        if order == "auto":
            order =self.nCollocation
    
        #use midpoint rule if uniform weighting
        if self.spacing == "uniform":
            weights = np.ones(self.collocationPoints.shape)*(self.bounds[1]-self.bounds[0])/self.nCollocation
        elif self.spacing == "legendre":
            # Note that leggaus output tuple where first is sampling points and second is sampling weights
            points = (np.polynomial.legendre.leggauss(order)[0]+1)*(self.bounds[1]-self.bounds[0])/2+self.bounds[0]
            weights = np.polynomial.legendre.leggauss(order)[1]*(self.bounds[1]-self.bounds[0])/2
        else:
            raise(Exception("Invalid collocation spacing used"))

        integral=f(points[0])*weights[0]

        #print("         Integral up to point ",0, ": ", integral)
        #integral=0.0
        for iPoint in range(1,points.size):
            integral+= f(points[iPoint])*weights[iPoint]
            #print("         Integral up to point ",iPoint, ": ", integral)
            
            
        return integral
