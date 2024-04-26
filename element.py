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
    _order=""
    _bounds=""
    _spacing=""
    _collocationPoints=""
    _interpolationPoints=""
    _basisCoeff=""
    
    
    def __init__(self,order=1,bounds =[0,1],spacing="uniform"):
        self.order=order
        self.bounds=bounds
        self.spacing=spacing
        
        self.setCollocationPoints()
        self.solveBasisCoeff()
        
        
        
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
            if value.size!=self.order or value.ndim!=1:
                raise Exception("Numpy array of non 1D with size order entered for collocationPoints: " + str(value))
            else:
                self._collocationPoints=value
        elif type(value)==list:
            if len(value)!=self.order:
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
            if value.size!=self.order+2 or value.ndim!=1:
                raise Exception("Numpy array of non 1D with size order+2 entered for interpolationPoints: " + str(value))
            else:
                self._interpolationPoints=value
        elif type(value)==list:
            if len(value)!=self.order+2:
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
        elif value.shape[0]!=self.order+2 or value.shape[1]!=self.order+2:
            raise Exception("Numpy array of non order+2 by order+2 entered for basisCoeff: " + str(value))
        else:
            self._basisCoeff=value

                
    
    #Write mappings between [-1,1] and element boundaries
    
    def mapToElementBounds(self,points):
        return self.bounds[0]+(points+1)*(self.bounds[1]-self.bounds[0])/2
    
    def mapFromElementBounds(self,points):
        return 1+2*(points-self.bounds[0])/(self.bounds[1]-self.bounds[0])
    
    #Setup Collocation Points
    def setCollocationPoints(self):
        if self.spacing=="uniform":
            self.collocationPoints=np.linspace(self.bounds[0],self.bounds[1],num=self.order+2)[1:self.order+1]
        elif self.spacing=="legendre":
            if self.order==1:
                self.collocationPoints=self.mapToElementBounds(np.array([0]))
            if self.order==2:
                self.collocationPoints=self.mapToElementBounds(np.array([-0.577350269189626,0.577350269189626]))
            if self.order==3:
                self.collocationPoints=self.mapToElementBounds(np.array([-0.774596669241483,0,0.774596669241483]))
            if self.order==4:
                self.collocationPoints=self.mapToElementBounds(np.array([-.86113631159405,
                                                                         -0.33998104358485,
                                                                          0.33998104358485,
                                                                          0.86113631159405]))
            if self.order==5:
                self.collocationPoints=self.mapToElementBounds(np.array([-0.90617984593866,
                                                                         -0.538469310105683,
                                                                          0,
                                                                          0.538469310105683,
                                                                          0.90617984593866]))
        interpolationPoints = np.empty((self.order+2))
        interpolationPoints[0]=self.bounds[0]
        interpolationPoints[1:-1]=self.collocationPoints
        interpolationPoints[-1]=self.bounds[1]
        self.interpolationPoints=interpolationPoints
            
    def solveBasisCoeff(self):
        basisCoeff=np.empty((self.order+2,self.order+2))
        massMatrix=np.empty((self.order+2,self.order+2))
        for iOrder in np.arange(self.order+2):
            massMatrix[:,iOrder]=self.interpolationPoints**iOrder
        invertedMass = np.linalg.inv(massMatrix)
        for iBasis in np.arange(self.order+2):
            b=np.zeros((self.order+2))
            b[iBasis]=1
            basisCoeff[iBasis]=np.dot(invertedMass,b)
        self.basisCoeff=basisCoeff
        
    def basisFunctions(self,x):
        if type(x)!=np.ndarray:
            basisFunctions=np.empty((self.order+2,1))
        elif x.ndim!=1:
                raise Exception("Multi-dimensional array entered for x: " + str(x))
        else :
            basisFunctions=np.empty((self.order+2,x.size))
        
        xExpanded = np.ones((self.order+2,)+x.shape)
        for iOrder in np.arange(self.order+2):
            xExpanded[iOrder]=x**iOrder
            
        for iBasis in np.arange(self.order+2):
            basisFunctions[iBasis] = np.sum(xExpanded.transpose()*self.basisCoeff[iBasis],axis=1)
                    
        return basisFunctions
    
    def basisFirstDeriv(self,x):
        if type(x)!=np.ndarray:
            basisFirstDeriv=np.empty((self.order+2,1))
        elif x.ndim!=1:
                raise Exception("Multi-dimensional array entered for x: " + str(x))
        else :
            basisFirstDeriv=np.empty((self.order+2,x.size))
        
        xExpanded = np.ones((self.order+1,)+x.shape)
        for iOrder in np.arange(self.order+1):
            xExpanded[iOrder]=(iOrder+1)*(x**iOrder)
            
        for iBasis in np.arange(self.order+2):
            basisFirstDeriv[iBasis] = np.sum(xExpanded.transpose()*self.basisCoeff[iBasis][1:],axis=1)
                    
        return basisFirstDeriv
    
    def basisSecondDeriv(self,x):
        if type(x)!=np.ndarray:
            basisSecondDeriv=np.empty((self.order+2,1))
        elif x.ndim!=1:
                raise Exception("Multi-dimensional array entered for x: " + str(x))
        else :
            basisSecondDeriv=np.empty((self.order+2,x.size))
        
        xExpanded = np.ones((self.order,)+x.shape)
        for iOrder in np.arange(self.order):
            xExpanded[iOrder]=(iOrder+2)*(iOrder+1)*(x**iOrder)
            
        for iBasis in np.arange(self.order+2):
            basisSecondDeriv[iBasis] = np.sum(xExpanded.transpose()*self.basisCoeff[iBasis][2:],axis=1)
        
        return basisSecondDeriv