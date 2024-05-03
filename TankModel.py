#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   FEM/TankModel.py
@Time    :   2024/02/09 11:16:26
@Author  :   Harley Hanes 
@Version :   1.0
@Contact :   hhanes@ncsu.edu
@License :   (C)Copyright 2024, Harley Hanes
@Desc    :   Class definition for the Orthogonal Collocation on Finite Elements solver of the adiabatic tank model
'''

import numpy as np
from CollocationElement import Element

class TankModel:
    _nElements=""
    _spacing=""
    _nCollocation=""
    _bounds=""
    _elements=""
    _firstOrderMat=""
    _secondOrderMat=""
    _massBoundaryMatInv=""
    _tempBoundaryMatInv=""
    _params=""
    
        
    def __init__(self,nElements=5, nCollocation=7, bounds=[0,1], spacing = "uniform",
                 params={"PeM": 0, "PeT": 0, "f": 0, "Le": 0, "Da": 0, "Beta": 0, "gamma": 0, "delta": 0}):
        self.nElements=nElements
        self.nCollocation=nCollocation
        self.bounds=bounds
        self.spacing=spacing
        self.params=params
        self.__makeElements__()
        self.__computeCollocationMatrices__()

    def __makeElements__(self):
        elements=[]
        print(self.bounds)
        print(self.nElements)
        bounds = np.linspace(self.bounds[0],self.bounds[1],num=self.nElements+1)
        for iElement in range(self.nElements):
            elements.append(Element(nCollocation = self.nCollocation,
                                   bounds=[bounds[iElement],bounds[iElement+1]],
                                   spacing=self.spacing
                                ))
        self.elements=elements
        
    def __computeCollocationMatrices__(self):
        firstOrderMat=np.zeros([self.nCollocation*self.nElements, (self.nCollocation+1)*self.nElements+1])
        secondOrderMat=np.zeros([self.nCollocation*self.nElements, (self.nCollocation+1)*self.nElements+1])
        for iRow in range(self.nCollocation*self.nElements):
            element = iRow // (self.nCollocation)
            collocation = iRow % (self.nCollocation)
            # Update iCol to only search through the the relevant columns 
            for iCol in np.arange(element*(self.nCollocation+1),(element+1)*(self.nCollocation+2)-element):
                basis = (iCol+element) % (self.nCollocation+2)
                firstOrderMat[iRow,iCol] = self.elements[element].basisFirstDeriv(self.elements[element].interpolationPoints[collocation])[basis]
                secondOrderMat[iRow,iCol] = self.elements[element].basisSecondDeriv(self.elements[element].interpolationPoints[collocation])[basis]
        self.firstOrderMat=firstOrderMat
        self.secondOrderMat=secondOrderMat
        
        massBoundaryMat=np.zeros([self.nElements+1,self.nElements+1])
        tempBoundaryMat=np.zeros([self.nElements+1,self.nElements+1])
        # Enter First and Last Rows Manually
        #massBoundaryMat[0,0]=1
        #massBoundaryMat[0,0]=2
        massBoundaryMat[0,0]=self.params["PeM"]*self.elements[0].basisFirstDeriv(self.bounds[0])[0]
        massBoundaryMat[0,1]=-self.elements[0].basisFirstDeriv(self.bounds[0])[-1]
        massBoundaryMat[-1,-2]=self.elements[-1].basisFirstDeriv(self.bounds[1])[0]
        massBoundaryMat[-1,-1]=self.elements[-1].basisFirstDeriv(self.bounds[1])[-1]
        
        tempBoundaryMat[0,0]=self.params["PeT"]*self.elements[0].basisFirstDeriv(self.elements[0].interpolationPoints[0])[0]
        tempBoundaryMat[0,1]=-self.elements[0].basisFirstDeriv(self.elements[0].interpolationPoints[0])[-1]
        tempBoundaryMat[0,-1]=self.params["f"]
        tempBoundaryMat[-1,-2]=self.elements[-1].basisFirstDeriv(self.bounds[1])[0]
        tempBoundaryMat[-1,-1]=self.elements[-1].basisFirstDeriv(self.bounds[1])[-1]
        # tempBoundaryMat[0,-1]=self.params["f"]
        
        #Enter interior rows
        for iRow in np.arange(1,self.nElements):
                massBoundaryMat[iRow,iRow-1]=self.elements[iRow-1].basisFirstDeriv(self.elements[iRow].interpolationPoints[0])[0]
                massBoundaryMat[iRow,iRow]=self.elements[iRow-1].basisFirstDeriv(self.elements[iRow].interpolationPoints[0])[-1] \
                                            -self.elements[iRow].basisFirstDeriv(self.elements[iRow].interpolationPoints[0])[0]
                massBoundaryMat[iRow,iRow+1]=-self.elements[iRow].basisFirstDeriv(self.elements[iRow].interpolationPoints[0])[-1]
                
                tempBoundaryMat[iRow,iRow-1]=self.elements[iRow-1].basisFirstDeriv(self.elements[iRow].interpolationPoints[0])[0]
                tempBoundaryMat[iRow,iRow]=self.elements[iRow-1].basisFirstDeriv(self.elements[iRow].interpolationPoints[0])[-1] \
                                            -self.elements[iRow].basisFirstDeriv(self.elements[iRow].interpolationPoints[0])[0]
                tempBoundaryMat[iRow,iRow+1]=-self.elements[iRow].basisFirstDeriv(self.elements[iRow].interpolationPoints[0])[-1]
        
        #Print Condition Numbers
        print("Mass Boundary Condition Number: " + str(np.linalg.cond(massBoundaryMat)))
        print("Temp Boundary Condition Number: " + str(np.linalg.cond(tempBoundaryMat)))
        #Compute inverse of Boundary mat''
        self.massBoundaryMatInv=np.linalg.inv(massBoundaryMat)
        self.tempBoundaryMatInv=np.linalg.inv(tempBoundaryMat)
    
    @property
    def nElements(self):
        return self._nElements
    @nElements.setter
    def nElements(self,value):
        if not isinstance(value,int):
            raise Exception("Non-int value entered for nElements")
        elif value <= 0:
            raise Exception("Non-positive integer entered for nElements")
        else:
            self._nElements=value
            
    @property
    def params(self):
        return self._params
    @params.setter
    def params(self,value):
        if type(value)!=dict:
            raise Exception("Non-dictionary value entered for nElements")
        
        if not "PeM" in value:
            raise Exception("PeM not in params dictionary")
        elif type(value["PeM"])!= float and type(value["PeM"])!= int:
            raise Exception("Non-numerical value entered for PeM: " + str(value["PeM"]))
        elif value["PeM"] < 0:
            raise Exception("Negative value eneter for PeM: " + str(value["PeM"]))
        
        if not "PeT" in value:
            raise Exception("PeT not in params dictionary")
        elif type(value["PeT"])!= float and type(value["PeT"])!= int:
            raise Exception("Non-numerical value entered for PeT: " + str(value["PeT"]))
        elif value["PeT"] < 0:
            raise Exception("Negative value eneter for PeT: " + str(value["PeT"]))
        
        if not "f" in value:
            raise Exception("f not in params dictionary")
        elif type(value["f"])!= float and type(value["f"])!= int:
            raise Exception("Non-numerical value entered for f: " + str(value["f"]))
        elif value["f"] < 0:
            raise Exception("Negative value eneter for f: " + str(value["f"]))
        
        
        if not "Le" in value:
            raise Exception("Le not in params dictionary")
        elif type(value["Le"])!= float and type(value["Le"])!= int:
            raise Exception("Non-numerical value entered for Le: " + str(value["Le"]))
        elif value["Le"] < 0:
            raise Exception("Negative value eneter for Le: " + str(value["Le"]))
        
        if not "Da" in value:
            raise Exception("Da not in params dictionary")
        elif type(value["Da"])!= float and type(value["Da"])!= int:
            raise Exception("Non-numerical value entered for Da: " + str(value["Da"]))
        elif value["Da"] < 0:
            raise Exception("Negative value eneter for Da: " + str(value["Da"]))

        if not "Beta" in value:
            raise Exception("Beta not in params dictionary")
        elif type(value["Beta"])!= float and type(value["Beta"])!= int:
            raise Exception("Non-numerical value entered for Beta: " + str(value["Beta"]))
        elif value["Beta"] < 0:
            raise Exception("Negative value eneter for Beta: " + str(value["Beta"]))
        

        if not "gamma" in value:
            raise Exception("gamma not in params dictionary")
        elif type(value["gamma"])!= float and type(value["gamma"])!= int:
            raise Exception("Non-numerical value entered for gamma: " + str(value["gamma"]))
        elif value["gamma"] < 0:
            raise Exception("Negative value eneter for gamma: " + str(value["gamma"]))
        
        if not "delta" in value:
            raise Exception("delta not in params dictionary")
        elif type(value["delta"])!= float and type(value["delta"])!= int:
            raise Exception("Non-numerical value entered for delta: " + str(value["delta"]))
        elif value["delta"] < 0:
            raise Exception("Negative value eneter for delta: " + str(value["delta"]))
        self._params=value
    
    @property
    def nCollocation(self):
        return self._nCollocation
    @nCollocation.setter
    def nCollocation(self,value):
        if not isinstance(value,int):
            raise Exception("Non-int value entered for nCollocation")
        elif value <= 0:
            raise Exception("Non-positive integer entered for nCollocation")
        else:
            self._nCollocation=value
    
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
        elif type(value)==list:
            if len(value)!=2:
                raise Exception("List of length other than 2 entered for bounds: " + str(value))
        self._bounds=value
    
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
    def elements(self):
        return self._elements
    @elements.setter
    def elements(self,value):
        if not isinstance(value,list):
            raise Exception("Non-list value entered for elements: " + str(value))
        else:
            for iElement in range(len(value)):
                if not isinstance(value[iElement],Element):
                    raise Exception("Non-Element object entered for elements[" + str(iElement)+"]: " + str(value[iElement]))
        self._elements=value

    @property 
    def firstOrderMat(self):
        return self._firstOrderMat
    @firstOrderMat.setter
    def firstOrderMat(self,value):
        if type(value)!=np.ndarray:
            raise Exception("Non-numpy array enetered for firstOrderMat: " +str(value))
        elif value.shape!=(self.nElements*self.nCollocation, self.nElements*(self.nCollocation+1)+1):
            raise Exception("Matrix of incorrect size entered for firstOrderMat: firstOrderMat.size="+str(value.shape))
        else:
            self._firstOrderMat=value
    @property 
    def secondOrderMat(self):
        return self._secondOrderMat
    @secondOrderMat.setter
    def secondOrderMat(self,value):
        if type(value)!=np.ndarray:
            raise Exception("Non-numpy array enetered for secondOrderMat: " +str(value))
        elif value.shape!=(self.nElements*self.nCollocation,self.nElements*(self.nCollocation+1)+1):
            raise Exception("Matrix of incorrect size entered for secondOrderMat: secondOrderMat.size="+str(value.shape))
        else:
            self._secondOrderMat=value
            
    @property 
    def massBoundaryMatInv(self):
        return self._massBoundaryMatInv
    @massBoundaryMatInv.setter
    def massBoundaryMatInv(self,value):
        if type(value)!=np.ndarray:
            raise Exception("Non-numpy array enetered for massBoundaryMatInv: " +str(value))
        elif value.shape!=(self.nElements+1,self.nElements+1):
            raise Exception("Matrix of incorrect size entered for massBoundaryMatInv: massBoundaryMat.size="+str(value.shape))
        
        self._massBoundaryMatInv=value
            
    @property 
    def tempBoundaryMatInv(self):
        return self._tempBoundaryMatInv
    @tempBoundaryMatInv.setter
    def tempBoundaryMatInv(self,value):
        if type(value)!=np.ndarray:
            raise Exception("Non-numpy array enetered for tempBoundaryMatInv: " +str(value))
        elif value.shape!=(self.nElements+1,self.nElements+1):
            raise Exception("Matrix of incorrect size entered for tempBoundaryMatInv: tempBoundaryMat.size="+str(value.shape))
       
        self._tempBoundaryMatInv=value
        
    
    
