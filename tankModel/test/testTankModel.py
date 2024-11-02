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

trueMassBoundaryMat=np.array([[1+(3/2)/params["PeM"], -2/params["PeM"], 1/2/params["PeM"], 0 ,0],
                              [0, 1, 0, 0, 0],
                              [1/2, -2, 3, -2, 1/2],
                              [0, 0, 0, 1, 0],
                              [0, 0, 1/2, -2, 3/2]
                              ])
trueTempBoundaryMat=np.array([[1+(3/2)/params["PeT"], -2/params["PeT"], 1/2/params["PeT"], 0 ,-params["f"]],
                              [0, 1, 0, 0, 0],
                              [1/2, -2, 3, -2, 1/2],
                              [0, 0, 0, 1, 0],
                              [0, 0, 1/2, -2, 3/2]
                              ])

#Check Boundary matrices

assert(np.isclose(trueMassBoundaryMat, model.massBoundaryMat).all())
assert(np.isclose(trueTempBoundaryMat, model.tempBoundaryMat).all())
#Check first and 2nd order matrices

#Check Integration: Spatial Integration Only
nCollocation=2
nElements=2
model = TankModel(nCollocation=nCollocation,nElements=nElements,spacing="legendre",bounds=[-2,2],params=params)
f=lambda x: x**2+x+1
fint = 9+1/3
integral=model.integrateSpace(f)

nCollocation=3
nElements=2
model = TankModel(nCollocation=nCollocation,nElements=nElements,spacing="legendre",bounds=[0,2],params=params)
f=lambda x: x**3+x**2+x+1
fint = 10+2/3
integral=model.integrateSpace(f)
assert(np.isclose(fint,integral))

#Check Integration: Spatial at multiple points and temporal
nCollocation=2
nElements=2
model = TankModel(nCollocation=nCollocation,nElements=nElements,spacing="legendre",bounds=[-2,2],params=params)
f=lambda x: np.outer(np.array([0,1,2]),x**2+x+1).flatten()
fint = np.array([0,9+1/3, 18+2/3])
fTempint = 18+2/3
integral,integralSpace=model.integrate(f,np.array([0,1,2]))
assert(np.isclose(fTempint,integral))
assert(np.isclose(fint,integralSpace).all())

#Check Integration: Spatial and Temporal Integration
print("testTankModely.py passes")