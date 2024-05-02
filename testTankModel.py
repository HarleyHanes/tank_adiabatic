import numpy as np
from TankModel import TankModel

nCollocation=1
nElements=2
params={"PeM": 1, "PeT": 1, "f": 1, "Le": 0, "Da": 0, "Beta": 0, "gamma": 0, "delta": 0}
model = TankModel(nCollocation=nCollocation,nElements=nElements,spacing="legendre",bounds=[0,1],params=params)

print(TankModel.firstOrderMat)
print(TankModel.secondOrderMat)
print(TankModel.massBoundaryMatInv)
print(TankModel.tempBoundaryMatInv)