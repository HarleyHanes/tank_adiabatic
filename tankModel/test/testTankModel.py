import sys
import os 
current_script_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath(os.path.join(current_script_dir,'..','..'))
sys.path.append(grandparent_dir)
print(sys.path)

import numpy as np
from tankModel.TankModel import TankModel
import scipy

print("Running testTankModel.py")
nCollocation=1
nElements=2
params={"PeM": 1, "PeT": 1, "f": 1, "Le": 1, "Da": 1, "beta": 1, "gamma": 1,"delta": 1, "vH":1}
model = TankModel(nCollocation=nCollocation,nElements=nElements,spacing="legendre",bounds=[-2,2],params=params)

#print(TankModel.firstOrderMat)
#print(TankModel.secondOrderMat)

trueMassBoundaryMat=np.array([[-3/2-params["PeM"], 2 ,-1/2, 0 ,0],
                              [0, 1, 0, 0, 0],
                              [1/2, -2, 3, -2, 1/2],
                              [0, 0, 0, 1, 0],
                              [0, 0, 1/2, -2, 3/2]
                              ])
trueTempBoundaryMat=np.array([[-3/2-params["PeT"], 2 ,-1/2, 0 ,params["f"]],
                              [0, 1, 0, 0, 0],
                              [1/2, -2, 3, -2, 1/2],
                              [0, 0, 0, 1, 0],
                              [0, 0, 1/2, -2, 3/2]
                              ])

# print(model.tempBoundaryMat)
# print(np.round(model.tempBoundaryMatInv*100000000)/10000000)
# print(model.tempRHSmat)
assert(np.isclose(trueMassBoundaryMat, model.massBoundaryMat).all())
assert(np.isclose(trueTempBoundaryMat, model.tempBoundaryMat).all())

print(np.round(model.dydt(np.array([1/2,1/2,1/2,1/2]),0)*100000000)/10000000)

#Check Stability???
#y=scipy.integrate.odeint(model.dydt,np.array([1,2,3,4]),np.linspace(0,20,10))
#print(model.integrate(model.computeFullCoeff(y)[:,0:5]))
#print(model.integrate(model.computeFullCoeff(y)[:,5:]))

print("testTankModely.py passes")