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

import numpy as np
import matplotlib.pyplot as plt
import mms.tankMMS as tankMMS

#Test 1: Check MMS Solutions for 2nd and 3rd order cases are computed correctly

nCollocations = [1]
#I think there's an error with the higher
spatialOrders=[2,3]  #Must be greater than 2 to satisfy BC
nElems = [2,4]  #Cant use nElems=1 due to some dimensionality issues with squeeze
#Note: Parameters can be any positive value except f=1
params={"PeM": 1, "PeT": 1, "f": 2, "Le": 1, "Da": 1, "beta": 1, "gamma": 1,"delta": 1, "vH":1}
tEval = np.linspace(0,3,100)
xEval = np.linspace(0,1,100)
error, solutions=tankMMS.runMMStest(spatialOrders,nCollocations,nElems,xEval,tEval,params)

linearCoeff=-2
uConstant=linearCoeff/params["PeM"]
vConstant=1/(1-params["f"])*(params["f"]+linearCoeff*(params["f"]+1/params["PeT"]))
#u for 2nd order case
assert(np.isclose(np.sum(np.abs(solutions[0,0,0,0,0,0,-1,:]-(xEval**2+linearCoeff*xEval+uConstant))),0))
#v for 2nd order case
assert(np.isclose(np.sum(np.abs(solutions[0,0,0,0,1,0,-1,:]-(xEval**2+linearCoeff*xEval+vConstant))),0))
print("2nd Order case passing")

linearCoeff=-2-3
uConstant=linearCoeff/params["PeM"]
vConstant=1/(1-params["f"])*(2*params["f"]+linearCoeff*(params["f"]+1/params["PeT"]))
#u for 2nd order case
assert(np.isclose(np.sum(np.abs(solutions[0,0,0,1,0,0,-1,:]-(xEval**3+xEval**2+linearCoeff*xEval+uConstant))),0))
#v for 2nd order case
assert(np.isclose(np.sum(np.abs(solutions[0,0,0,1,1,0,-1,:]-(xEval**3+xEval**2+linearCoeff*xEval+vConstant))),0))
print("3nd Order case passing")