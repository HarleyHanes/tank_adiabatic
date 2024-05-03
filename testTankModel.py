import numpy as np
from TankModel import TankModel

nCollocation=1
nElements=2
params={"PeM": 1, "PeT": 1, "f": 1, "Le": 0, "Da": 0, "Beta": 0, "gamma": 0, "delta": 0}
model = TankModel(nCollocation=nCollocation,nElements=nElements,spacing="legendre",bounds=[-2,2],params=params)

#print(TankModel.firstOrderMat)
#print(TankModel.secondOrderMat)
print(TankModel.massBoundaryMatInv)
print(TankModel.tempBoundaryMatInv)

trueMassBoundaryMat=np.array([[-3/2*params["PeM"], -7/2, 0],
                              [1/2, 3, 1/2],
                              [0, 1/2, 3/2]])

trueTempBoundaryMat=np.array([[-3/2*params["PeT"], -7/2, params["f"]],
                              [1/2, 3, 1/2],
                              [0, 1/2, 3/2]])

print(np.linalg.inv(trueMassBoundaryMat))
print(np.linalg.inv(trueTempBoundaryMat))
assert(np.isclose(np.linalg.inv(trueMassBoundaryMat), TankModel.massBoundaryMatInv).all())
assert(np.isclose(np.linalg.inv(trueTempBoundaryMat), TankModel.tempBoundaryMatInv).all())