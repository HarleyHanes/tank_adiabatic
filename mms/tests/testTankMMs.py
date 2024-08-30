#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   FEM/mms/tests/testTankMMS.py
@Time    :   2024/05/24 11:16:26
@Author  :   Harley Hanes 
@Version :   1.0
@Contact :   hhanes@ncsu.edu
@License :   (C)Copyright 2024, Harley Hanes
@Desc    :   Unit tests for tankMMS.py
'''
import sys
import os 
current_script_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath(os.path.join(current_script_dir,'..','..'))
sys.path.append(grandparent_dir)
print(sys.path)

import numpy as np
import matplotlib.pyplot as plt
import mms.tankMMS as tankMMS

print("Running testTankMMS.py")
#Test 1: Convergence Rate testing
error=np.array([1,1/2, (1/2)**3, (1/2)**6, (1/2)**10, (1/2)**15])
step_size=np.array([1,1/2,1/4,1/8,1/16,1/32])
convergenceRates=tankMMS.computeConvergenceRates(step_size,error)
assert(np.all(np.isclose(convergenceRates,np.array([1.0,2.0,3.0,4.0,5.0]))))
print("     Convergence Rate Test passes")



#Test 2: Check MMS Solutions for 2nd and 3rd order cases are computed correctly
nCollocations = [1]
#I think there's an error with the higher
spatialOrders=[2]  #Must be greater than or equal to 2 to satisfy BC
nElems = [2]  #Cant use nElems=1 due to some dimensionality issues with squeeze
#Note: Parameters can be any positive value except f=1
params={"PeM": 1, "PeT": 1, "f": 2, "Le": 1, "Da": 1, "beta": 1, "gamma": 1,"delta": 1, "vH":1}
tEval = np.linspace(0,3,50)
xEval = np.linspace(0,1,100)
error, solutions, convergenceRates=tankMMS.runMMStest(spatialOrders,nCollocations,nElems,xEval,tEval,params,verbosity=0)

print(solutions[0,0,0,0,0,0,-1,:])
print(solutions.shape)
linearCoeff=2
uConstant=linearCoeff/params["PeM"]
vConstant=1/(1-params["f"])*(params["f"]+linearCoeff*(params["f"]+1/params["PeT"]))
print((xEval**2+linearCoeff*xEval+uConstant))
#u for 2nd order case
assert(np.isclose(np.sum(np.abs(solutions[0,0,0,0,0,0,-1,:]-(xEval**2+linearCoeff*xEval+uConstant))),0))
#v for 2nd order case
assert(np.isclose(np.sum(np.abs(solutions[0,0,0,0,1,0,-1,:]-(xEval**2+linearCoeff*xEval+vConstant))),0))
print("     2nd Order case passing")

linearCoeff=-2-3
uConstant=linearCoeff/params["PeM"]
vConstant=1/(1-params["f"])*(2*params["f"]+linearCoeff*(params["f"]+1/params["PeT"]))
#u for 3nd order case
assert(np.isclose(np.sum(np.abs(solutions[0,0,0,1,0,0,-1,:]-(xEval**3+xEval**2+linearCoeff*xEval+uConstant))),0))
#v for 3nd order case
assert(np.isclose(np.sum(np.abs(solutions[0,0,0,1,1,0,-1,:]-(xEval**3+xEval**2+linearCoeff*xEval+vConstant))),0))
print("     3nd Order case passing")

print("testTankMMS.py passes")