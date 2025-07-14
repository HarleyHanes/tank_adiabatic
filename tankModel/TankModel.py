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
from collocationElement.CollocationElement import Element
from tankModel.romData import RomData

class TankModel:
    _verbosity=""
    _nElements=""
    _spacing=""
    _nCollocation=""
    _bounds=""
    _elements=""
    _collocationPoints=""
    _firstOrderMat=""
    _secondOrderMat=" "
    _massBoundaryMat=" "
    _tempBoundaryMat=" "
    _massBoundaryMatInv=" "
    _tempBoundaryMatInv=" "
    _params=" "
    _massRHSmat=""
    _tempRHSmat=""
    
    
    
    #===============================Getters and Setters======================================
    @property
    def verbosity(self):
        return self._verbosity
    @verbosity.setter
    def verbosity(self,value):
        if value in (0,"n","N"):
            self._verbosity=0
        elif value in (1,"m","M"):
            self._verbosity=1
        elif value in (2,"h","H"):
            self._verbosity=2
        elif value ==3:
            self._verbosity=3
        else:
            raise Exception("Invalid verbosity, use 0/n/N (none), 1/m/M (medium), 2/h/H (high), 3 (very high)")
 
    @property
    def nElements(self):
        return self._nElements
    @nElements.setter
    def nElements(self,value):
        if type(value)==np.ndarray:
            #Check that it's a 1D with size 1
            if value.size!=1:
                raise Exception("numpy array entered for nElements size: ", value.size)
            elif type(value[0])!=np.int64:
                raise Exception("Non-integer numpy array entered for nElements: ", value)
            elif value[0] <= 0:
                raise Exception("Non-positive integer entered for nElements: ",value)   
            else:
                self._nElements=int(value[0])
        elif type(value)==np.int64:
            if value <= 0:
                raise Exception("Non-positive integer entered for nElements: ",value) 
            else:
                self._nElements=int(value)
        elif isinstance(value,int):
            if value <= 0:
                raise Exception("Non-positive integer entered for nElements: ",value)
            else:
                self._nElements=value
        else:
            raise Exception("Unrecognized type entered for nElements: ", type(value))
 
    @property
    def nCollocation(self):
        return self._nCollocation
    @nCollocation.setter
    def nCollocation(self,value):
        if not isinstance(value,int):
            raise Exception("Non-int value entered for nCollocation: ", value)
        elif value <= 0:
            raise Exception("Non-positive integer entered for nCollocation: ", value)
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
    def collocationPoints(self):
        return self._collocationPoints
    @collocationPoints.setter
    def collocationPoints(self,value):
        if not isinstance(value,list) and not isinstance(value,np.ndarray):
            raise Exception("Non-list/ array value entered for elements: " + str(value))
        else:
            if isinstance(value,list):
                if len(value)!=self.nCollocation*self.nElements:
                    raise Exception("Collocation points of incorrect length entered: " + str(value))
            elif isinstance(value,np.ndarray):
                if value.ndim!=1:
                    raise Exception("Non-1D array entered for collocation points: " + str(value))
                elif value.size!=self.nCollocation*self.nElements:
                    raise Exception("Array with incorrect number of collocation points entered:" + str(value))
        self._collocationPoints=value

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
    def massBoundaryMat(self):
        return self._massBoundaryMat
    @massBoundaryMat.setter
    def massBoundaryMat(self,value):
        if type(value)!=np.ndarray:
            raise Exception("Non-numpy array enetered for massBoundaryMatInv: " +str(value))
        elif value.shape!=(self.nElements*(self.nCollocation+1)+1,self.nElements*(self.nCollocation+1)+1):
            raise Exception("Matrix of incorrect size entered for massBoundaryMatInv: massBoundaryMat.size="+str(value.shape))
        
        self._massBoundaryMat=value
            
    @property 
    def tempBoundaryMat(self):
        return self._tempBoundaryMat
    @tempBoundaryMat.setter
    def tempBoundaryMat(self,value):
        if type(value)!=np.ndarray:
            raise Exception("Non-numpy array enetered for tempBoundaryMatInv: " +str(value))
        elif value.shape!=(self.nElements*(self.nCollocation+1)+1,self.nElements*(self.nCollocation+1)+1):
            raise Exception("Matrix of incorrect size entered for tempBoundaryMat: tempBoundaryMat.size="+str(value.shape))
       
        self._tempBoundaryMat=value
    
    @property 
    def massBoundaryMatInv(self):
        return self._massBoundaryMatInv
    @massBoundaryMatInv.setter
    def massBoundaryMatInv(self,value):
        if type(value)!=np.ndarray:
            raise Exception("Non-numpy array enetered for massBoundaryMatInv: " +str(value))
        elif value.shape!=(self.nElements*(self.nCollocation+1)+1,self.nElements*(self.nCollocation+1)+1):
            raise Exception("Matrix of incorrect size entered for massBoundaryMatInv: massBoundaryMat.size="+str(value.shape))
        
        self._massBoundaryMatInv=value
            
    @property 
    def tempBoundaryMatInv(self):
        return self._tempBoundaryMatInv
    @tempBoundaryMatInv.setter
    def tempBoundaryMatInv(self,value):
        if type(value)!=np.ndarray:
            raise Exception("Non-numpy array enetered for tempBoundaryMatInv: " +str(value))
        elif value.shape!=(self.nElements*(self.nCollocation+1)+1,self.nElements*(self.nCollocation+1)+1):
            raise Exception("Matrix of incorrect size entered for tempBoundaryMatInv: tempBoundaryMat.size="+str(value.shape))
       
        self._tempBoundaryMatInv=value
        
    @property 
    def massRHSmat(self):
        return self._massRHSmat
    @massRHSmat.setter
    def massRHSmat(self,value):
        if type(value)!=np.ndarray:
            raise Exception("Non-numpy array enetered for tempRHSmat: " +str(value))
       # elif value.shape!=(self.nElements*(self.nCollocation),self.nElements*(self.nCollocation)):
            #raise Exception("Matrix of incorrect size entered for massRHSmat: massRHSmat.size="+str(value.shape))
        
        self._massRHSmat=value
            
    @property 
    def tempRHSmat(self):
        return self._tempRHSmat
    @tempRHSmat.setter
    def tempRHSmat(self,value):
        if type(value)!=np.ndarray:
            raise Exception("Non-numpy array enetered for tempRHSmat: " +str(value))
        #elif value.shape!=(self.nElements*(self.nCollocation),(self.nElements*(self.nCollocation))):
            #raise Exception("Matrix of incorrect size entered for tempRHSmat: tempRHSmat.size="+str(value.shape))
        
        self._tempRHSmat=value
        
               
    @property
    def params(self):
        return self._params
    @params.setter
    def params(self,value):
        if type(value)!=dict:
            raise Exception("Non-dictionary value entered for nElements")
        
        if not "PeM" in value:
            raise Exception("PeM not in params dictionary")
        elif type(value["PeM"])!= float and type(value["PeM"])!= int and type(value["PeM"])!= complex:
            raise Exception("Non-numerical value entered for PeM: " + str(value["PeM"]))
        
        if not "PeT" in value:
            raise Exception("PeT not in params dictionary")
        elif type(value["PeT"])!= float and type(value["PeT"])!= int and type(value["PeT"])!= complex:
            raise Exception("Non-numerical value entered for PeT: " + str(value["PeT"]))
       
        if not "f" in value:
            raise Exception("f not in params dictionary")
        elif type(value["f"])!= float and type(value["f"])!= int and type(value["f"])!= complex:
            raise Exception("Non-numerical value entered for f: " + str(value["f"]))
        
        
        if not "Le" in value:
            raise Exception("Le not in params dictionary")
        elif type(value["Le"])!= float and type(value["Le"])!= int and type(value["Le"])!= complex:
            raise Exception("Non-numerical value entered for Le: " + str(value["Le"]))
       
        if not "Da" in value:
            raise Exception("Da not in params dictionary")
        elif type(value["Da"])!= float and type(value["Da"])!= int and type(value["Da"])!= complex:
            raise Exception("Non-numerical value entered for Da: " + str(value["Da"]))
        
        if not "beta" in value:
            raise Exception("beta not in params dictionary")
        elif type(value["beta"])!= float and type(value["beta"])!= int and type(value["beta"])!= complex:
            raise Exception("Non-numerical value entered for beta: " + str(value["beta"]))
        

        if not "gamma" in value:
            raise Exception("gamma not in params dictionary")
        elif type(value["gamma"])!= float and type(value["gamma"])!= int and type(value["gamma"])!= complex:
            raise Exception("Non-numerical value entered for gamma: " + str(value["gamma"]))
        
        if not "delta" in value:
            raise Exception("delta not in params dictionary")
        elif type(value["delta"])!= float and type(value["delta"])!= int and type(value["delta"])!= complex:
            raise Exception("Non-numerical value entered for delta: " + str(value["delta"]))
        
        if not "vH" in value:
            raise Exception("vH not in params dictionary")
        elif type(value["vH"])!= float and type(value["vH"])!= int and type(value["vH"])!= complex:
            raise Exception("Non-numerical value entered for vH: " + str(value["vH"]))
        
        if not "PeM-boundary" in value:
            value["PeM-boundary"]=value["PeM"]
        elif type(value["PeM-boundary"])!= float and type(value["PeM-boundary"])!= int and type(value["PeM-boundary"])!= complex:
            raise Exception("Non-numerical value entered for PeM-boundary: " + str(value["PeM-boundary"]))
        
        if not "PeT-boundary" in value:
            value["PeT-boundary"]=value["PeT"]
        elif type(value["PeT-boundary"])!= float and type(value["PeT-boundary"])!= int and type(value["PeT-boundary"])!= complex:
            raise Exception("Non-numerical value entered for PeT-boundary: " + str(value["PeT-boundary"]))
        
        self._params=value
    
        
    #================================Object Formation Functions=============================================
        
    def __init__(self,nElements=5, nCollocation=7, bounds=[0,1], spacing = "uniform",
                 params={"PeM": 1, "PeT": 1, "f": 0, "Le": 1, "Da": 0, "beta": 0, "gamma": 0, "delta": 0, "vH": 0},verbosity=0):
        self.verbosity=verbosity
        self.nElements=nElements
        self.nCollocation=nCollocation
        self.bounds=bounds
        self.spacing=spacing
        self.params=params
        self.__makeElements__()
        self.__computeCollocationMatrices__()
    

    def __makeElements__(self):
        elements=[]
        bounds = np.linspace(self.bounds[0],self.bounds[1],num=self.nElements+1)
        for iElement in range(self.nElements):
            element = Element(nCollocation = self.nCollocation,
                                   bounds=[bounds[iElement],bounds[iElement+1]],
                                   spacing=self.spacing
                                )
            elements.append(element)
            if iElement==0:
                collocationPoints = element.collocationPoints
            else :
                collocationPoints = np.concatenate((collocationPoints,element.collocationPoints),axis=0)
        self.elements=elements
        self.collocationPoints=collocationPoints

        
    def __computeCollocationMatrices__(self):
        firstOrderMat=np.zeros([self.nCollocation*self.nElements, (self.nCollocation+1)*self.nElements+1])
        secondOrderMat=np.zeros([self.nCollocation*self.nElements, (self.nCollocation+1)*self.nElements+1])
        for iRow in range(self.nCollocation*self.nElements):
            element = iRow // (self.nCollocation)
            collocation = iRow % (self.nCollocation)
            # Update iCol to only search through the the relevant columns 
            for iCol in np.arange(element*(self.nCollocation+1),(element+1)*(self.nCollocation+2)-element):
                basis = (iCol+element) % (self.nCollocation+2)
                firstOrderMat[iRow,iCol] = self.elements[element].basisFirstDeriv(self.elements[element].collocationPoints[collocation])[basis]
                secondOrderMat[iRow,iCol] = self.elements[element].basisSecondDeriv(self.elements[element].collocationPoints[collocation])[basis]
        self.firstOrderMat=firstOrderMat
        self.secondOrderMat=secondOrderMat
        
        massBoundaryMat=np.eye((self.nElements*(self.nCollocation+1)+1))
        tempBoundaryMat=np.eye((self.nElements*(self.nCollocation+1)+1))
        #Enter interior rows
        for iRow in np.arange(0,self.nElements*(self.nCollocation+1)+1,self.nCollocation+1):
            #Left BC
            if iRow==0:
                massBoundaryMat[0,0:self.nCollocation+2]=self.elements[0].basisFirstDeriv(self.bounds[0])
                massBoundaryMat[0,0]-=self.params["PeM-boundary"]
                tempBoundaryMat[0,0:self.nCollocation+2]=self.elements[0].basisFirstDeriv(self.bounds[0])
                tempBoundaryMat[0,0]-=self.params["PeT-boundary"]
                tempBoundaryMat[0,-1]=self.params["f"]*self.params["PeT-boundary"]
            #Right BC
            elif iRow==self.nElements*(self.nCollocation+1):
                massBoundaryMat[iRow,-(self.nCollocation+2):]=self.elements[-1].basisFirstDeriv(self.bounds[1])
                tempBoundaryMat[iRow,-(self.nCollocation+2):]=self.elements[-1].basisFirstDeriv(self.bounds[1])
            #Internal Boundaries
            else:
                element= iRow //(self.nCollocation+1)
                #interalBoundary = iRow // (self.nCollocation)
                tempBoundaryMat[iRow,((element-1)*(self.nCollocation+1)):(element*(self.nCollocation+1)+1)] \
                    =self.elements[element-1].basisFirstDeriv(self.elements[element].interpolationPoints[0])
                tempBoundaryMat[iRow,(element*(self.nCollocation+1)):((element+1)*(self.nCollocation+1)+1)] \
                    -=self.elements[element].basisFirstDeriv(self.elements[element].interpolationPoints[0])
                massBoundaryMat[iRow,((element-1)*(self.nCollocation+1)):(element*(self.nCollocation+1)+1)] \
                    =self.elements[element-1].basisFirstDeriv(self.elements[element].interpolationPoints[0])
                massBoundaryMat[iRow,(element*(self.nCollocation+1)):((element+1)*(self.nCollocation+1)+1)] \
                    -=self.elements[element].basisFirstDeriv(self.elements[element].interpolationPoints[0])
        self.massBoundaryMat=massBoundaryMat
        self.tempBoundaryMat=tempBoundaryMat
        pointExpansionMat=np.zeros((self.nElements*(self.nCollocation+1)+1,self.nElements*self.nCollocation))
        for i in range(self.nElements):
            rowStart=i*(self.nCollocation+1)+1
            colStart=i*self.nCollocation
            pointExpansionMat[rowStart:rowStart+self.nCollocation,colStart:colStart+self.nCollocation]=np.eye(self.nCollocation)

        self.pointExpansionMat=pointExpansionMat
        self.massFullCoeffMat=np.linalg.solve(massBoundaryMat,pointExpansionMat)
        self.tempFullCoeffMat=np.linalg.solve(tempBoundaryMat,pointExpansionMat)
        if self.verbosity>1:
            if self.verbosity>2:
                print("Point Expansion Matrix")
                print(pointExpansionMat)
            print("Mass Boundary Condition Number: ", np.linalg.cond(massBoundaryMat))
            print("Temp Boundary Condition Number: ", np.linalg.cond(tempBoundaryMat))
            if self.verbosity >2:
                print("Mass Boundary Matrix")
                print(massBoundaryMat)
                print("Temp Boundary Matrix")
                print(tempBoundaryMat)
            print("Mass Closure Condition Number: " + str(np.linalg.cond(self.massFullCoeffMat)))
            print("Temp Closure Condition Number: " + str(np.linalg.cond(self.tempFullCoeffMat)))
            if self.verbosity >2:
                print("Mass Closure  Matrix")
                print(self.massFullCoeffMat)
                print("Temp Closure Matrix")
                print(self.tempFullCoeffMat)
            print("1st Order Matrix Condition Number: " + str(np.linalg.cond(self.firstOrderMat)))
            print("2nd Order Matrix Condition Number: " + str(np.linalg.cond(self.secondOrderMat)))
            if self.verbosity >2:
                print("1st Order Matrix")
                print(self.firstOrderMat)
                print("2nd Order Matrix")
                print(self.secondOrderMat)
            print("Mass Full Domain Condition Number: " + str(np.linalg.cond(-self.firstOrderMat+1/self.params["PeM"]*self.secondOrderMat)))
            print("Temp Full Domain Condition Number: " + str(np.linalg.cond((-self.firstOrderMat+1/self.params["PeT"]*self.secondOrderMat)/self.params["Le"])))
            if self.verbosity >2:
                print("Mass Full Domain  Matrix")
                print(-self.firstOrderMat+1/self.params["PeM"]*self.secondOrderMat)
                print("Temp Full Domain Matrix")
                print((-self.firstOrderMat+1/self.params["PeT"]*self.secondOrderMat)/self.params["Le"])
        self.massRHSmat = np.matmul(-self.firstOrderMat+1/self.params["PeM"]*self.secondOrderMat,self.massFullCoeffMat)
        self.tempRHSmat = np.matmul(-self.firstOrderMat+1/self.params["PeT"]*self.secondOrderMat,self.tempFullCoeffMat)
        if self.verbosity>1:
            print("Mass RHS Condition Number: " + str(np.linalg.cond(self.massRHSmat)))
            print("Temp RHS Condition Number: " + str(np.linalg.cond(self.tempRHSmat)))
            if self.verbosity >2:
                print("Mass RHS  Matrix")
                print(self.massRHSmat)
                print("Temp RHS Matrix")
                print(self.tempRHSmat)
        
        #Sensitivity Matrices
        doublePointExpansionMat=np.append(pointExpansionMat,np.zeros(pointExpansionMat.shape),axis=1)
        doublePointExpansionMat=np.append(doublePointExpansionMat,np.append(np.zeros(pointExpansionMat.shape),pointExpansionMat,axis=1),axis=0)
        dudPeMboundaryMat=np.zeros((2*((self.nCollocation+1)*self.nElements+1),2*((self.nCollocation+1)*self.nElements+1)))
        dudPeMboundaryMat[:(self.nCollocation+1)*self.nElements+1,:(self.nCollocation+1)*self.nElements+1]=self.massBoundaryMat
        dudPeMboundaryMat[(self.nCollocation+1)*self.nElements+1:,(self.nCollocation+1)*self.nElements+1:]=self.massBoundaryMat
        dudPeMboundaryMat[(self.nCollocation+1)*self.nElements+1,0]=-1
        self.dudPeMboundaryMat=dudPeMboundaryMat
        dudPeMfullCoeffMat=np.linalg.solve(dudPeMboundaryMat,doublePointExpansionMat)[(self.nCollocation+1)*self.nElements+1:,]
        self.dudPeMrhsMat=np.matmul(-self.firstOrderMat+1/self.params["PeM"]*self.secondOrderMat,dudPeMfullCoeffMat)
        self.dudPeMSecondOrderMat=np.matmul(self.secondOrderMat/(self.params["PeM"]**2),self.massFullCoeffMat)

        dvdPeTboundaryMat=np.zeros((2*((self.nCollocation+1)*self.nElements+1),2*((self.nCollocation+1)*self.nElements+1)))
        dvdPeTboundaryMat[:(self.nCollocation+1)*self.nElements+1,:(self.nCollocation+1)*self.nElements+1]=self.tempBoundaryMat
        dvdPeTboundaryMat[(self.nCollocation+1)*self.nElements+1:,(self.nCollocation+1)*self.nElements+1:]=self.tempBoundaryMat
        dvdPeTboundaryMat[(self.nCollocation+1)*self.nElements+1,0]=-1
        dvdPeTboundaryMat[(self.nCollocation+1)*self.nElements+1,(self.nCollocation+1)*self.nElements]=self.params["f"]
        self.dvdPeTboundaryMat=dvdPeTboundaryMat
        dvdPeTfullCoeffMat=np.linalg.solve(dvdPeTboundaryMat,doublePointExpansionMat)[(self.nCollocation+1)*self.nElements+1:,]
        self.dvdPeTrhsMat=np.matmul(-self.firstOrderMat+1/self.params["PeT"]*self.secondOrderMat,dvdPeTfullCoeffMat)
        self.dvdPeTSecondOrderMat=np.matmul(self.secondOrderMat/(self.params["PeT"]**2),self.tempFullCoeffMat)

        dvdfboundaryMat=np.zeros((2*((self.nCollocation+1)*self.nElements+1),2*((self.nCollocation+1)*self.nElements+1)))
        dvdfboundaryMat[:(self.nCollocation+1)*self.nElements+1,:(self.nCollocation+1)*self.nElements+1]=self.tempBoundaryMat
        dvdfboundaryMat[(self.nCollocation+1)*self.nElements+1:,(self.nCollocation+1)*self.nElements+1:]=self.tempBoundaryMat
        dvdfboundaryMat[(self.nCollocation+1)*self.nElements+1,(self.nCollocation+1)*self.nElements]=self.params["PeT"]
        self.dvdfboundaryMat=dvdfboundaryMat
        dvdffullCoeffMat=np.linalg.solve(dvdfboundaryMat,doublePointExpansionMat)[(self.nCollocation+1)*self.nElements+1:,]
        self.dvdfrhsMat=np.matmul(-self.firstOrderMat+1/self.params["PeT"]*self.secondOrderMat,dvdffullCoeffMat)
        
    
    #================================Eval Functions==========================================
    
    def dydt(self,y,t):
        #Construct expanded y with boundary points
        u=y[0:self.nElements*self.nCollocation]
        v=y[self.nElements*self.nCollocation:]
        # dydt=np.append(
        #         np.dot(self.massRHSmat,np.linalg.solve(self.massBoundaryMat,np.dot(self.pointExpansionMat,u)))+self.params["Da"]*(1-u)*np.exp(self.params["gamma"]*self.params["beta"]*v/(1+self.params["beta"]*v)),
        #         np.dot(self.tempRHSmat,np.linalg.solve(self.tempBoundaryMat,np.dot(self.pointExpansionMat,v)))+(self.params["Da"]*(1-u)*np.exp(self.params["gamma"]*self.params["beta"]*v/(1+self.params["beta"]*v))
        #                                    +self.params["delta"]*(self.params["vH"]-v))/self.params["Le"])
        dydt=np.append(
                 np.dot(self.massRHSmat,u)+self.params["Da"]*(1-u)*np.exp(self.params["gamma"]*self.params["beta"]*v/(1+self.params["beta"]*v)),
                 (np.dot(self.tempRHSmat,v)+self.params["Da"]*(1-u)*np.exp(self.params["gamma"]*self.params["beta"]*v/(1+self.params["beta"]*v))
                                            +self.params["delta"]*(self.params["vH"]-v))/self.params["Le"])
        
        return dydt
    
    def dydtSource(self,y,t,source):
        return self.dydt(y,t) + source(t)
    
    def dydtSens(self,y,t,paramSelect=[]):
        """Defines the differential equation for sensitivity equations"""
        nPoints=self.nElements*self.nCollocation
        u=y[0:nPoints]
        v=y[nPoints:2*nPoints]
        dydt=self.dydt(y[0:2*nPoints],t)
        dvdt=dydt[nPoints:]
        eqCounter=2
        for param in paramSelect:
            dudParam = y[eqCounter*nPoints:(eqCounter+1)*nPoints]
            dvdParam = y[(eqCounter+1)*nPoints:(eqCounter+2)*nPoints]
                #Linear Piece
            match param:
                case "PeM":
                    uCombined = np.append(u,dudParam)
                    ddudParamdt=np.dot(self.dudPeMrhsMat,uCombined)-np.dot(self.dudPeMSecondOrderMat,u)
                    ddvdParamdt=np.dot(self.tempRHSmat,dvdParam)
                case "PeT":
                    vCombined = np.append(v,dvdParam)
                    ddudParamdt=np.dot(self.massRHSmat,dudParam)
                    ddvdParamdt=np.dot(self.dvdPeTrhsMat,vCombined)-np.dot(self.dvdPeTSecondOrderMat,v)
                case "f":
                    vCombined = np.append(v,dvdParam)
                    ddudParamdt=np.dot(self.massRHSmat,dudParam)
                    ddvdParamdt=np.dot(self.dvdfrhsMat,vCombined)
                case _:
                    ddudParamdt=np.dot(self.massRHSmat,dudParam)
                    ddvdParamdt=np.dot(self.tempRHSmat,dvdParam)
            #Linear Source Piece
            match param:
                case "delta":
                    linearTerm = (self.params["vH"]-v-self.params["delta"]*dvdParam)
                case "vH":
                    linearTerm = self.params["delta"]*(1-dvdParam)
                case "Le":
                    linearTerm = -dvdt-self.params["delta"]*dvdParam
                case _:
                    linearTerm = -self.params["delta"]*dvdParam
            ddvdParamdt+=linearTerm
            #Nonlinear Source Piece
            match param:
                case "Da":
                    nonLinearTerm = np.exp(self.params["gamma"]*self.params["beta"]*v/(1+self.params["beta"]*v))*\
                                        (1-u+self.params["Da"]*(-dudParam+(1-u)*self.params["gamma"]*self.params["beta"]*dvdParam/(1+self.params["beta"]*v)**2))
                case "beta":
                    nonLinearTerm = self.params["Da"]*np.exp(self.params["gamma"]*self.params["beta"]*v/(1+self.params["beta"]*v))*\
                                        (-dudParam+(1-u)*self.params["gamma"]*(v+self.params["beta"]*dvdParam)/(1+self.params["beta"]*v)**2)
                case "gamma":
                    nonLinearTerm = self.params["Da"]*np.exp(self.params["gamma"]*self.params["beta"]*v/(1+self.params["beta"]*v))*\
                                        (-dudParam+(1-u)*self.params["beta"]*(v+self.params["beta"]*(v**2)+self.params["gamma"]*dvdParam)/(1+self.params["beta"]*v)**2)
                case _:
                    nonLinearTerm = self.params["Da"]*np.exp(self.params["gamma"]*self.params["beta"]*v/(1+self.params["beta"]*v))*\
                                        (-dudParam+(1-u)*self.params["gamma"]*self.params["beta"]*dvdParam/(1+self.params["beta"]*v)**2)
            ddudParamdt+=nonLinearTerm
            ddvdParamdt+=nonLinearTerm
            dydt=np.append(dydt,ddudParamdt)
            dydt=np.append(dydt,ddvdParamdt/self.params["Le"])
            eqCounter+=2
        return dydt
    

    def eval(self,xEval,modelCoeff, output="full",deriv=0):
        """eval computes the value of u and v at every point in xEval given the collocation element expression provided by modelCoeff"""
        if output not in ("full","seperated","u","v"):
            raise(Exception("Invalid entry for output used, must be full, seperated, u, or v"))
        #Convert xEval to numpy array if not already so logical operators will work on its indices
        if type(xEval)!=np.ndarray:
            xEval=np.array(xEval)
        #Get model coeff including the boundary coeffeceints
        uFull,vFull = self.computeFullCoeff(modelCoeff,output="seperated")

        #Initialize u and v values at each x point and snapshot
        if modelCoeff.ndim==1:
            uEval=np.zeros(xEval.shape)
            vEval=np.zeros(xEval.shape)
        elif modelCoeff.ndim==2:
            uEval=np.zeros((modelCoeff.shape[0],)+xEval.shape)
            vEval=np.zeros((modelCoeff.shape[0],)+xEval.shape)

        if self.verbosity > 1:
            print("uFull shape: ", uFull.shape)
        #Compute u and v values within each element
        # print("uFull shape: ", uFull.shape)
        # print("uEval shape: ", uEval.shape)
        for iElement in range(self.nElements):
            element=self.elements[iElement]
            #Get x values for just the interior points of selected element. Use .4 here (but any number between .25 and .5 would work) to mark points that are strickly 
            xElementIndices =  (element.bounds[0]<= xEval)&(element.bounds[1] >= xEval)
            # print(xElementIndices)
            xElement=xEval[xElementIndices]
            #Compute the values of the basis polynomials at each x location
            if deriv==0:
                basisValues = element.basisFunctions(xElement)
            elif deriv==1:
                basisValues = element.basisFirstDeriv(xElement)
            elif deriv==2:
                basisValues = element.basisSecondDeriv(xElement)

            
            if modelCoeff.ndim==1:
                uEval[xElementIndices]=np.dot(uFull[iElement*(self.nCollocation+1):(iElement+1)*(self.nCollocation+1)+1],basisValues)
                vEval[xElementIndices]=np.dot(vFull[iElement*(self.nCollocation+1):(iElement+1)*(self.nCollocation+1)+1],basisValues)
            #Don't need to check that uFull and vFull have only 1 or 2 dimensions since check occurs in computeFullCoeff
            else :
                # print("uFull for Element " , iElement, ": ", uFull[0,iElement*(self.nCollocation+1):(iElement+1)*(self.nCollocation+1)+1])
                # print("basisValues for Element " , iElement, ": ", basisValues)
                uEval[:,xElementIndices]=np.dot(uFull[:,iElement*(self.nCollocation+1):(iElement+1)*(self.nCollocation+1)+1],basisValues)
                vEval[:,xElementIndices]=np.dot(vFull[:,iElement*(self.nCollocation+1):(iElement+1)*(self.nCollocation+1)+1],basisValues)
            # print("Computed uEval values for element ", iElement, ": ", np.dot(uFull[:,iElement*(self.nCollocation+1):(iElement+1)*(self.nCollocation+1)+1],basisValues)[0,:])
            # print("uEval at after element ", iElement, ": ", uEval[-1,:])
        if output=="seperated":
            return uEval, vEval
        elif output =="u":
            return uEval
        elif output =="v":
            return vEval
        else:
            #Recombine uEval and vEval into a single set
            return np.concatenate((uEval,vEval),axis=-1)

    
    def computeFullCoeff(self,collocationCoeff,output="full"):
        """computeFullCoeff computes the coeffecients including the boundary coeff for each element from the interior coeffecients.
                This is done by applying the closure relations encoded in the *FullCoeffMat"""
        #Multiple snapshot case, snapshots must be in first-dimension
        if collocationCoeff.ndim == 2:
            #Seperate collocation coeff into the u and v components
            u=collocationCoeff[:,0:self.nElements*self.nCollocation]
            v=collocationCoeff[:,self.nElements*self.nCollocation:]
            uFull=np.matmul(self.massFullCoeffMat,u.transpose()).transpose()
            vFull=np.matmul(self.tempFullCoeffMat,v.transpose()).transpose()
            if output=="seperated":
                return uFull,vFull
            elif output == "u":
                return uFull
            elif output == "v":
                return vFull
            elif output == "full":
                return np.concatenate((uFull,vFull),axis=1)
            else:
                raise Exception("Invalid output return option entered")
        #Sinlge snapshot case
        elif collocationCoeff.ndim ==1:
            u=collocationCoeff[0:self.nElements*self.nCollocation]
            v=collocationCoeff[self.nElements*self.nCollocation:]
            if output=="seperated":
                return np.dot(self.massFullCoeffMat,u),np.dot(self.tempFullCoeffMat,v)
            elif output=="u":
                return np.dot(self.massFullCoeffMat,u)
            elif output=="v":
                return np.dot(self.tempfullCoeffMat,v)
            elif output=="full": 
                return np.append(np.dot(self.massFullCoeffMat,u),np.dot(self.tempFullCoeffMat,v))
        else:
            raise Exception("Invalid dimension entered for collocationCoeff: " + str(collocationCoeff.ndim))
        
    def integrateSpace(self,f,order="auto"):
        integralSpace=self.elements[0].integrate(f, order =order)
        #print("     Integral up to element 0: ", integralSpace)
        for i in range(1,self.nElements):
            #print(self.elements[i].integrate(f))
            integralSpace+= self.elements[i].integrate(f, order=order)
            #print("     Integral up to element ",i, ": ", integralSpace)
        return integralSpace
    
    def integrate(self,f,tValues,integrateTime=True,order="auto"):
        #SHould implement check that tValues are equally spaced
        #Use the Legendre quadrature methods to compute the spatial integrals at each time value
        #integralSpace=np.zeros(tValues.shape)
        # for iT in range(tValues.size):
        #     fT=lambda x:f(x)[iT]
        #     for i in range(self.nElements):
        #         print(self.elements[i].integrate(fT))
        #         integralSpace[iT]+= self.elements[i].integrate(fT)
        integralSpace = self.integrateSpace(f, order=order)

        if integrateTime:
            #Use a trapezoid method to compute the integrals in time
            #Note: Using adjustment to rule that doesn't require equally spaced points
            deltaT=tValues[1:]-tValues[:-1]
            weights=np.zeros(tValues.shape)
            weights[0]=deltaT[0]/2
            weights[-1]=deltaT[-1]/2
            weights[1:-1]=(deltaT[:-1]+deltaT[1:])/2
            integral = np.sum(integralSpace*weights)
            return integral, integralSpace
        else:
            return integralSpace
    

    #================================POD-ROM===================================================

    def constructPodRom(self,modelCoeff,x,W,modeThreshold,nonlinDim = "max",mean="mean",useEnergyThreshold=True,adjustModePairs=False,nDeimPoints="none"):
        #Get Snapshots and derivatives of snapshots
        uEval,vEval = self.eval(x,modelCoeff,output="seperated")
        #Compute Mean, keeping dimension so casting works
        if mean =="mean":
            meanCoeff = np.mean(modelCoeff,axis=0)
            uMean,vMean = self.eval(x,meanCoeff,output="seperated")
            uMeanx,vMeanx = self.eval(x,meanCoeff,output="seperated",deriv=1)
            uMeanxx,vMeanxx = self.eval(x,meanCoeff,output="seperated",deriv=2)
        elif mean =="zero":
            uMean = np.zeros(uEval.shape[1])
            vMean = np.zeros(vEval.shape[1])
            uMeanx = np.zeros(uEval.shape[1])
            vMeanx = np.zeros(vEval.shape[1])
            uMeanxx = np.zeros(uEval.shape[1])
            vMeanxx = np.zeros(vEval.shape[1])
        elif mean =="ones":
            uMean = np.ones(uEval.shape[1])
            vMean = np.ones(vEval.shape[1])
            uMeanx = np.zeros(uEval.shape[1])
            vMeanx = np.zeros(vEval.shape[1])
            uMeanxx = np.zeros(uEval.shape[1])
            vMeanxx = np.zeros(vEval.shape[1])
        elif mean =="first":
            uMean,vMean = self.eval(x,modelCoeff[0],output="seperated")
            uMeanx,vMeanx = self.eval(x,modelCoeff[0],output="seperated",deriv=1)
            uMeanxx,vMeanxx = self.eval(x,modelCoeff[0],output="seperated",deriv=2)
        elif mean==np.ndarray:
            if mean.ndim==1:
                if mean.size==self.collocationPoints.size:
                    uMean,vMean = self.eval(x,mean,output="seperated")
                    uMeanx,vMeanx = self.eval(x,mean,output="seperated",deriv=1)
                    uMeanxx,vMeanxx = self.eval(x,mean,output="seperated",deriv=2)
                else:
                    raise ValueError("Invalid size for mean")
            else:
                raise ValueError("Invalid dimension for mean")

        uEval-=uMean
        vEval-=vMean
        uEvalx,vEvalx = self.eval(x,modelCoeff, output="seperated", deriv =1)
        uEvalx-=uMeanx
        vEvalx-=vMeanx

        uEvalxx,vEvalxx = self.eval(x,modelCoeff, output="seperated", deriv =2)
        uEvalxx-=uMeanxx
        vEvalxx-=vMeanxx
        if np.isclose(modelCoeff[:,:self.nElements*self.nCollocation],modelCoeff[:,self.nElements*self.nCollocation:], atol=1e-10).all():
            print("u and v coeffecients are the same")
            if not np.isclose(uEval,vEval, atol=1e-10).all():
                raise ValueError("u and v coeffecients are the same but evaluations are not")
            if not np.isclose(uEvalx,vEvalx, atol=1e-10).all():
                raise ValueError("u and v coeffecients are the same but derivatives are not")
            if not np.isclose(uEvalxx,vEvalxx, atol=1e-10).all():
                raise ValueError("u and v coeffecients are the same but 2nd derivatives are not")
        uModes, uModesx, uModesxx, uTimeModes, uTruncationError, uSingularValues =self.computePODmodes(W, uEval.transpose(),uEvalx.transpose(),uEvalxx.transpose(),modeThreshold,useEnergyThreshold=useEnergyThreshold,adjustModePairs=adjustModePairs)
        vModes, vModesx, vModesxx, vTimeModes, vTruncationError, vSingularValues =self.computePODmodes(W, vEval.transpose(),vEvalx.transpose(),vEvalxx.transpose(),modeThreshold,useEnergyThreshold=useEnergyThreshold,adjustModePairs=adjustModePairs)
        if np.isclose(modelCoeff[:,:self.nElements*self.nCollocation],modelCoeff[:,self.nElements*self.nCollocation:], atol=1e-10).all():
            print("L2 Difference in Modes: ", np.sqrt(np.sum(W@(uModes-vModes)**2)))
            print("Linf Difference in Modes: ", np.max(np.abs(uModes-vModes)))
        if self.bounds[0]==x[0]:
            for i in range(uModes.shape[1]):
                if not np.isclose(uModes[0,i]-uModesx[0,i]/self.params["PeM-boundary"],0,atol=1e-6):
                    print("Left Boundary Condition not satisfied for uMode", i, "by: ", uModes[0,i]-uModesx[0,i]/self.params["PeM"])
                if not np.isclose(uModesx[-1,i],0,atol=1e-6):
                    print("Right Boundary Condition not satisfied for uMode", i, "by: ", uModesx[-1,i])
            for i in range(vModes.shape[1]):
                if not np.isclose(vModes[0,i]-vModesx[0,i]/self.params["PeT-boundary"]-vModes[-1,i]*self.params["f"],0,atol=1e-6):
                    print("Left Boundary Condition not satisfied for vMode", i, "by: ", vModes[0,i]-vModesx[0,i]/self.params["PeM"]-vModes[-1,i]*self.params["f"])
                if not np.isclose(vModesx[-1,i],0,atol=1e-6):
                    print("Right Boundary Condition not satisfied for vMode", i, "by: ", vModesx[-1,i])
        uModesWeighted, uModesInt, uRomMassMean, uRomFirstOrderMat, uRomFirstOrderMean, uRomSecondOrderMat, uRomSecondOrderMean\
              = self.computeRomMatrices(W,uMean, uMeanx, uMeanxx, uModes, uModesx,uModesxx)
        vModesWeighted, vModesInt, vRomMassMean, vRomFirstOrderMat, vRomFirstOrderMean, vRomSecondOrderMat, vRomSecondOrderMean\
              = self.computeRomMatrices(W,vMean, vMeanx, vMeanxx, vModes, vModesx,vModesxx)
        truncationError=np.mean([uTruncationError,vTruncationError])
        
        if nonlinDim=="max":
            uNonlinDim = uModes.shape[1]
            vNonlinDim = vModes.shape[1]
        elif nonlinDim<1:
            uNonlinDim = int(np.ceil(uModes.shape[1]*nonlinDim))
            vNonlinDim = int(np.ceil(vModes.shape[1]*nonlinDim))
        elif nonlinDim>1:
            uNonlinDim = nonlinDim
            vNonlinDim = nonlinDim
        else:
            raise ValueError("Unrecognized value for nonlinDim: ", nonlinDim)
        
        #Compute DEIM Projection
        # If any string input is entered for nDEIMpoints assume not using DEIM

        if type(nDeimPoints) == str:
            deimProjection = np.eye(uModes.shape[0])
            uNonLinProjection = uModesWeighted.transpose()
            vNonLinProjection = vModesWeighted.transpose()
        else:
            u = uModes@uTimeModes.transpose()+uMean.reshape((x.size,1))
            v = vModes@vTimeModes.transpose()+vMean.reshape((x.size,1))
            nonLinData = (1-u)*np.exp(self.params["gamma"]*self.params["beta"]\
                                        *v/(1+self.params["beta"]*v))
            deimBasis,deimProjection = self.computeDEIMbasis(nonLinData,nDeimPoints)
            uNonLinProjection = self.computeDEIMmatrices(uModesWeighted,deimBasis,deimProjection)
            vNonLinProjection = self.computeDEIMmatrices(vModesWeighted,deimBasis,deimProjection)
            
        return RomData(x, W, uTimeModes, uMean, uModes, uModesx, uModesxx, uModesWeighted,
                        uModesInt, uRomMassMean, uRomFirstOrderMat, uRomFirstOrderMean,
                        uRomSecondOrderMat, uRomSecondOrderMean, vTimeModes, vMean,
                        vModes, vModesx, vModesxx, vModesWeighted, vModesInt,
                        vRomMassMean, vRomFirstOrderMat, vRomFirstOrderMean,
                        vRomSecondOrderMat, vRomSecondOrderMean,uSingularValues,vSingularValues, uNonlinDim,vNonlinDim,deimProjection,uNonLinProjection,vNonLinProjection), truncationError

    def computeDEIMbasis(self,nonLinData,nDeimModes):
        #Compute basis for non-linear evaluationd data using POD
        Psi,sigma,V = np.linalg.svd(nonLinData)
        # Initialize DEIM Algorithm
        deimBasis = Psi[:,[0]]
        P=np.eye(Psi.shape[0])[:,[np.argmax(np.abs(Psi[:,-1]))]]
        # Loop through DEIM modes
        for i in range(1,nDeimModes):
            #Solve for nonlin Coeff
            q = np.linalg.solve(P.transpose()@deimBasis,P.transpose()@Psi[:,i] )
            #Compute residual
            r=Psi[:,i]-deimBasis@q
            #Update DEIM matrices
            P=np.append(P,np.eye(Psi.shape[0])[:,[np.argmax(np.abs(r))]],axis=1)
            deimBasis = np.append(deimBasis,Psi[:,[i]],axis=1)

        return deimBasis, P

    def computeDEIMmatrices(self,podBasisWeighted,deimBasis, P):
        #System is A=(PhiW)'Psi(P'Psi)^(-1), to do standard linear solve need to transfrom it to (Psi'P)A'=Psi'(PhiW)
        deimProjection = np.linalg.solve(deimBasis.transpose()@P, deimBasis.transpose()@podBasisWeighted).transpose()
        #Note: Not currently computing projection because we're still computing projection at each step to simplify dydt implementation,
        #       If want to improve in future will need to compute this here and pass unweighted POD modes as well.
        #podProjection = P.transpose()@podBasis
        
        return deimProjection

    
    def computePODmodes(self,W, snapshots, snapshotsx, snapshotsxx,modeThreshold,useEnergyThreshold=True,adjustModePairs=False,groupSeperations="null"):
        # INCOMPLETE: Scale each observed response to [0,1] for POD so that sensitivities with large values aren't weighted more, may need setting up this change in TankModel
        # if type(groupSeperations)==np.ndarray:
        #     raise ValueError("groupSeperations for mixed inputs incomplete")
        #     # Compute norm of each snapshot
        #     norms = np.sqrt(np.sum(W@(snapshots**2),axis=0))
        #     snapshotScaling = np.zeros(norms.shape)
        #     # Preallocate average norms for each group of snapshots
        #     groupNorms = np.zeros(groupSeperations.size+1)
        #     groupSeperations = np.append(groupSeperations,0)
        #     groupSeperations = np.append(groupSeperations,snapshots.shape[1])
        #     for iGroup in range(groupSeperations.size):
        #         groupNorms[iGroup] = np.mean(norms[groupSeperations[iGroup]:groupSeperations[iGroup+1]])
        #         snapshotScaling[groupSeperations[iGroup]:groupSeperations[iGroup+1]] = 1/groupNorms[iGroup]
        #     # Relative snapshots to average norm
        #     snapshotsRel = snapshots/snapshotScaling
        # elif groupSeperations!="null":
        #     raise ValueError("Invalid groupSeperations input")

        #Use transpose of evals since eval is time x space but standard snapshot matrix is space x time
        if np.isclose(W/W[0,0],np.eye(W.shape[0])).all():
            modes, S, timeModes = np.linalg.svd(snapshots*np.sqrt(W[0,0]),full_matrices=False)
            timeModes=timeModes.transpose()
        else:
            #Get eigen decomp of UtWU
            timeModes,S,null= np.linalg.svd(snapshots.transpose()@W@snapshots,full_matrices=False)
            #Check symmetry of eigen decomp
            if not np.isclose(timeModes@S,S@null).all():
                print("WARNING: Singular value scaled time eigen decomp not symmetric")
                print("Error: ", np.sqrt(np.sum(np.sum((timeModes@S-S@null.transpose())**2))/np.sum(np.sum((timeModes@S)**2))))
                print("W: ", W)
                print("timeModes-timeModesT: ",timeModes-null.transpose())
            #Have to take squareroot of S for scaling
            S=np.sqrt(S)
        modes = snapshots@timeModes@np.diag(1/S)
        
        print("Minimum Singular Value: ", np.min(S))
        #Compute modes for derivatives
        modesx = snapshotsx @ timeModes @ (np.diag(1/S))
        modesxx = snapshotsxx @ timeModes @ (np.diag(1/S))

        #Rescale time-modes by average norms
        if useEnergyThreshold:
            #Create threshold for S
            totalEnergy=np.sum(S)
            cumulEnergy=0
            nModes=0
            while (cumulEnergy<modeThreshold)and(nModes<S.size):
                nModes+=1
                cumulEnergy=np.sum(S[:nModes])/totalEnergy
        else:
            totalEnergy=np.sum(S)
            nModes=modeThreshold
            cumulEnergy=np.sum(S[:nModes])/totalEnergy
        if adjustModePairs and nModes>1:
            #print(S[:nModes+1])
            #Approach 1: Compare distance in singular values
            #adjacent_distance = S[nModes-1:nModes+1]-S[nModes-2:nModes]
            #Appraoch 2: Compare innerproducts between last mode and derivative of next and preceeding mode
            adjacent_distance = np.abs(np.array([modes[:,nModes-1].transpose()@W@modesx[:,nModes-2],modes[:,nModes-1].transpose()@W@modesx[:,nModes]]))
            #If S[nModes-1] is closer to S[nModes] than S[nModes-2], add another mode
            #if adjacent_distance[1]<adjacent_distance[0]:
            #If inner product of nMode with excluded mode deriv is greater than with included mode, add excluded mode
            if adjacent_distance[1]>adjacent_distance[0]:
                nModes+=1
                cumulEnergy=np.sum(S[:nModes])/totalEnergy
        print("Cumulative Energy of Modes: ", cumulEnergy)
        print("Number of modes used: ", nModes)
        modes = modes[:,:nModes]
        modesx = modesx[:,:nModes]
        modesxx = modesxx[:,:nModes]
        timeModes = (timeModes@np.diag(S))[:,:nModes]
        #Check Orthonormality of modes
        if not np.isclose(modes.transpose()@W@modes,np.eye(modes.shape[1])).all():
            print("WARNING: Modes not orthonormal")
            print("Phi^TWPhi = ", modes.transpose()@W@modes)
            print("Departure from Orthonormality: ", np.sum(np.eye(modes.shape[1])-modes.transpose()@W@modes))
        #Check the POD decomposition is accurate in FOM space
        podError = np.sqrt(np.sum(W@((snapshots-modes@timeModes.transpose())**2))/np.sum(W@(snapshots**2)))
        podIcError = np.sqrt(np.sum(W@((snapshots[:,0]-(modes@timeModes.transpose())[:,0])**2))/np.sum(W@(snapshots[:,0]**2)))
        print("POD Relative Error: ", podError)
        print("POD IC Relative Error: ", podIcError)
        return modes, modesx, modesxx, timeModes, podError, S[:nModes]

    def computeRomMatrices(self,W,mean,  meanx, meanxx, podModes, podModesx,podModesxx):
        podModesWeighted = W @ podModes
        podModesInt = np.sum(podModesWeighted.transpose(),axis=1) 
        romMassMean = podModesWeighted.transpose() @ mean
        romFirstOrderMat = podModesWeighted.transpose() @ podModesx
        romFirstOrderMean = podModesWeighted.transpose() @ meanx
        romSecondOrderMat = podModesWeighted.transpose() @ podModesxx
        romSecondOrderMean = podModesWeighted.transpose() @ meanxx
        return podModesWeighted, podModesInt, romMassMean, romFirstOrderMat, romFirstOrderMean, romSecondOrderMat, romSecondOrderMean

    def getQuadWeights(self,nPoints,quadRule):
        if quadRule == "simpson" :
            if nPoints%2==0:
                raise ValueError("Simpson's rule requires an odd number of points")
            else:
                x=np.linspace(self.bounds[0],self.bounds[1],nPoints)
                #Get quadrature weights using simpson's rule
                w=np.ones(x.size)/3*(self.bounds[1]-self.bounds[0])/(x.size-1)
                w[1:x.size-1:2]*=4
                if np.size(x)>=5:
                    w[2:x.size-2:2]*=2
        elif quadRule == "uniform":
            x=np.linspace(self.bounds[0],self.bounds[1],nPoints)
            w=np.ones(np.size(x))/((x.size)*(self.bounds[1]-self.bounds[0]))
        elif quadRule == "monte carlo":
            x=np.array([self.bounds[0]])
            x=np.append(x,np.sort(np.random.sample(nPoints-2))*(self.bounds[1]-self.bounds[0])+self.bounds[0],axis=0)
            x=np.append(x,np.array([self.bounds[1]]),axis=0)
            w=np.ones(np.size(x))/(x.size)*(self.bounds[1]-self.bounds[0])
        elif quadRule == "gauss-legendre":
            x,w= np.polynomial.legendre.leggauss(nPoints)
            x=(x+1)*(self.bounds[1]-self.bounds[0])/2+self.bounds[0]
            w=w*(self.bounds[1]-self.bounds[0])/2
        elif quadRule == "gauss-legendre adjusted":
            x,w= np.polynomial.legendre.leggauss(nPoints)
            x=(x+1)*(self.bounds[1]-self.bounds[0])/2+self.bounds[0]
            x[0]=self.bounds[0]
            x[-1]=self.bounds[1]
            w=np.ones(np.size(x))/(x.size)*(self.bounds[1]-self.bounds[0])
        else:
            raise ValueError("Invalid quadRule")
        return x, np.diag(w)

    def dydtPodRom(self,y,t,romData,paramSelect=[],penaltyStrength=0):
        if type(paramSelect)==str:
            paramSelect=[paramSelect]
        u=y[0:romData.uNmodes]
        v=y[romData.uNmodes:romData.uNmodes+romData.vNmodes]
        #Note: This step below is not optimal for computation time, keeping it currently because it simplifies implementation for
        #       for switching DEIM on and off. Optimal would be pre-computing P^T@modes and P^T@mean for u and v
        uNonlin=romData.deimProjection@(romData.uModes[:,:romData.uNonlinDim]@u[:romData.uNonlinDim]+romData.uMean)
        vNonlin=romData.deimProjection@(romData.vModes[:,:romData.vNonlinDim]@v[:romData.vNonlinDim]+romData.vMean)
        uFull=romData.uModes@u+romData.uMean
        vFull=romData.vModes@v+romData.vMean
        uFullx=romData.uModesx@u+romData.uMean
        vFullx=romData.vModesx@v+romData.vMean
        # dudt=(romData.uRomSecondOrderMat/self.params["PeM"]- romData.uRomFirstOrderMat)@u\
        #         +romData.uRomSecondOrderMean/self.params["PeM"]-romData.uRomFirstOrderMean    
        # dvdt=((romData.vRomSecondOrderMat/self.params["PeT"]-romData.vRomFirstOrderMat)@v\
        #         +romData.vRomSecondOrderMean/self.params["PeT"]-romData.vRomFirstOrderMean)/self.params["Le"]
        # dvdt=((romData.vRomSecondOrderMat/self.params["PeT"]-romData.vRomFirstOrderMat)@v\
        #         +romData.vRomSecondOrderMean/self.params["PeT"]-romData.vRomFirstOrderMean\
        #         +self.params["delta"]*(self.params["vH"]*romData.vModesInt-v-romData.vRomMassMean)\
        #         +self.params["Da"]*(romData.vModesInt - (romData.vModesWeighted.transpose()@romData.uModes)@u))/self.params["Le"]
        dudt=(romData.uRomSecondOrderMat/self.params["PeM"]- romData.uRomFirstOrderMat)@u\
                +romData.uRomSecondOrderMean/self.params["PeM"]-romData.uRomFirstOrderMean\
                +self.params["Da"]*romData.uNonLinProjection\
                                    @((1-uNonlin)*np.exp(self.params["gamma"]*self.params["beta"]\
                                      *vNonlin/(1+self.params["beta"]*vNonlin)))
        dvdt=((romData.vRomSecondOrderMat/self.params["PeT"]-romData.vRomFirstOrderMat)@v\
                +romData.vRomSecondOrderMean/self.params["PeT"]-romData.vRomFirstOrderMean\
                +self.params["delta"]*(self.params["vH"]*romData.vModesInt-v-romData.vRomMassMean)\
                +self.params["Da"]*romData.vNonLinProjection
                                    @((1-uNonlin)*np.exp(self.params["gamma"]*self.params["beta"]\
                                        *vNonlin/(1+self.params["beta"]*vNonlin))))/self.params["Le"]
        #Boundary Penalty
        #u
        dudt -= penaltyStrength*romData.uModes[0,:]*(uFull[0]-uFullx[0]/self.params["PeM"])
        dudt -= penaltyStrength*romData.uModes[-1,:]*uFull[-1]
        #v
        dvdt -= penaltyStrength*romData.vModes[0,:]*(vFull[0]-vFullx[0]/self.params["PeM"]-self.params["f"]*vFull[-1])
        dvdt -= penaltyStrength*romData.vModes[-1,:]*vFullx[-1]
        dydt = np.append(dudt,dvdt)
        eqCounter=1
        for param in paramSelect:
            dudParam = y[eqCounter*(romData.uNmodes+romData.vNmodes):eqCounter*(romData.uNmodes+romData.vNmodes)+romData.uNmodes]
            dvdParam = y[eqCounter*(romData.uNmodes+romData.vNmodes)+romData.uNmodes:(eqCounter+1)*(romData.uNmodes+romData.vNmodes)]
            dudParamFull=np.matmul(romData.uModes,dudParam)+romData.uMean
            dvdParamFull=np.matmul(romData.vModes,dvdParam)+romData.vMean
            #Define Advection/ Diffusion terms
            ddudParamdt=(romData.uRomSecondOrderMat /self.params["PeM"]-romData.uRomFirstOrderMat)@ dudParam\
                            +romData.uRomSecondOrderMean/self.params["PeM"]-romData.uRomFirstOrderMean
            ddvdParamdt=(romData.vRomSecondOrderMat /self.params["PeT"]-romData.vRomFirstOrderMat)@ dvdParam\
                            +romData.vRomSecondOrderMean/self.params["PeT"]-romData.vRomFirstOrderMean
            if param=="PeM":
                ddudParamdt+=-(romData.uRomSecondOrderMat@u+romData.uRomSecondOrderMean)/(self.params["PeM"]**2)
            elif param=="PeT":
                ddvdParamdt+=-(romData.vRomSecondOrderMat@v+romData.vRomSecondOrderMean)/(self.params["PeT"]**2)
            #Construct Additional Linear terms
            if param=="vH":
                ddvdParamdt+=(romData.vModesInt-dvdParam)*self.params["delta"] - self.params["delta"]*romData.vRomMassMean
            elif param=="delta":
                ddvdParamdt+=self.params["vH"]*romData.vModesInt-v-self.params["delta"]*dvdParam - self.params["delta"]*romData.vRomMassMean
            elif param=="Le":
                ddvdParamdt+= -dvdt-self.params["delta"]*dvdParam - self.params["delta"]*romData.vRomMassMean
            else:
                ddvdParamdt+= -self.params["delta"]*dvdParam - self.params["delta"]*romData.vRomMassMean



            # Construct nonlinear term
            if param in ["vH", "delta","Le","PeM","PeT","f"]:
                nonlinearTerm=self.params["Da"]*np.exp(self.params["gamma"]*self.params["beta"]\
                                                *vFull/(1+self.params["beta"]*vFull))\
                                                *((1-uFull)*(self.params["gamma"]*self.params["beta"]\
                                                           *dvdParamFull/((1+self.params["beta"]*vFull)**2))\
                                                - dudParamFull)
            elif param=="Da":
                nonlinearTerm=(self.params["Da"] *((1-uFull)*(self.params["gamma"]*self.params["beta"]\
                                                           *dvdParamFull/((1+self.params["beta"]*vFull)**2))\
                                                - dudParamFull)\
                             +(1-uFull))*np.exp(self.params["gamma"]*self.params["beta"]\
                                             *vFull/(1+self.params["beta"]*vFull))
            elif param=="beta":
                nonlinearTerm=self.params["Da"]*np.exp(self.params["gamma"]*self.params["beta"]\
                                                       *vFull/(1+self.params["beta"]*vFull))
                nonlinearTerm*=((1-uFull)*self.params["gamma"]*(vFull+self.params["beta"]*dvdParamFull)\
                                                                /((1+self.params["beta"]*vFull)**2)\
                                 - dudParamFull)
            elif param=="gamma":
                nonlinearTerm=self.params["Da"]*np.exp(self.params["gamma"]*self.params["beta"]\
                                                       *vFull/(1+self.params["beta"]*vFull))
                nonlinearTerm*=((1-uFull)*self.params["beta"]*(vFull+self.params["beta"]*(vFull**2)+self.params["gamma"]*dvdParamFull)\
                                                                /((1+self.params["beta"]*vFull)**2)\
                                 - dudParamFull)
            else:
                raise(Exception("Invalid param value: "+param))            
            ddudParamdt+=romData.uModesWeighted.transpose() @ nonlinearTerm
            ddvdParamdt+=romData.vModesWeighted.transpose() @ nonlinearTerm
            #Construct boundary term
            #NOTE: I think there are currently errors in this, check the BP formulation and then confirm this is implemented correctly
            # dudParamxLeftBoundary=np.dot(romData.uModesx[0,:],dudParam)+romData.uMean[0]
            # dvdParamxLeftBoundary=np.dot(romData.vModesx[0,:],dvdParam)+romData.vMean[0]
            # if param in ["vH", "delta", "Le", "Da","beta","gamma"]:
            #     ddudParamdt += penaltyStrength*(dudParamFull[0]-dudParamxLeftBoundary/self.params["PeM"])
            #     ddvdParamdt += penaltyStrength*(dvdParamFull[0]-(dvdParamxLeftBoundary/self.params["PeT"]+self.params["f"]*dvdParamFull[-1]))
            # elif param=="f":
            #     ddudParamdt += penaltyStrength*(dudParamFull[0]-dudParamxLeftBoundary/self.params["PeM"])
            #     ddvdParamdt += penaltyStrength*(dvdParamFull[0]-(dvdParamxLeftBoundary/self.params["PeT"]+self.params["f"]*dvdParamFull[-1]+vFull[-1]))
            # elif param=="PeM":
            #     uxLeftBoundary=np.dot(romData.uModesx[0,:],dudParam)+romData.uMean[0]
            #     ddudParamdt += penaltyStrength*(dudParamFull[0]-dudParamxLeftBoundary/self.params["PeM"]+uxLeftBoundary/(self.params["PeM"]**2))
            #     ddvdParamdt += penaltyStrength*(dvdParamFull[0]-(dvdParamxLeftBoundary/self.params["PeT"]+self.params["f"]*dvdParamFull[-1]))
            # elif param=="PeT":
            #     vxLeftBoundary=np.dot(romData.vModesx[0,:],dvdParam)+romData.vMean[0]
            #     ddudParamdt += penaltyStrength*(dudParamFull[0]-dudParamxLeftBoundary/self.params["PeM"])
            #     ddvdParamdt += penaltyStrength*(dvdParamFull[0]-(dvdParamxLeftBoundary-vxLeftBoundary/self.params["PeT"])/self.params["PeT"]-self.params["f"]*dvdParamFull[-1])

            #Scale RHS of v by Le
            ddvdParamdt/=self.params["Le"]

            dydt=np.append(dydt,ddudParamdt)
            dydt=np.append(dydt,ddvdParamdt)
            eqCounter+=1
        #print("Step Completed for t=",t)
        return dydt
    
    def computeRomError(self,uEval,vEval,uRom,vRom, W,tPoints,norm="Linf"):
        #Map from romCoeff to rom Solution
        if norm == "L2" or norm==r"$L_2$":
            #Compute joint-error
            errorU = np.sqrt(np.sum(W @ (uEval-uRom)**2))/ np.sum(W @ (uEval)**2)
            errorV = np.sqrt(np.sum(W @ (vEval-vRom)**2))/ np.sum(W @ (vEval)**2)
            #error = np.sqrt(np.max(np.sum(W @(uEval-uRom)**2,axis=0)))#/np.sum((W @vEval)**2))
            #error = np.sqrt(np.max(np.sum(W @(vEval-vRom)**2,axis=0)))#/np.sum((W @vEval)**2))
        elif norm == "Linf" or norm==r"$L_\infty$":
            errorU = np.max(np.abs(uEval-uRom))
            errorV = np.max(np.abs(vEval-vRom))
            #error = np.max(np.abs(vEval-vRom))#/np.max(np.abs(vEval))
            # error = np.max(np.abs(uEval-uRom))#/np.max(np.abs(vEval))
        error = (errorU+errorV)/2
        return error
#Class that holds all the data defining a particular POD-ROM model. We define a seperate class to TankModel since
# a single FOM may have numerous different ROMs computed from it. Properties not common to all ROMs are stored in this class for easier function parsing
