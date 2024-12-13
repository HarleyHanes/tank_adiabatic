import sys
import os 
current_script_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath(os.path.join(current_script_dir,'..','..'))
sys.path.append(grandparent_dir)
print(sys.path)

import numpy as np
from tankModel.TankModel import TankModel
import scipy

#Test POD mode computation
params={"PeM": 300, "PeT": 100, "f": 0, "Le": 3, "Da": 0, "beta": 0, "gamma": 0,"delta": 2, "vH":1}
#Define nElem=1,nColl=2 model with roots at -1 and 1 for simplicity
model = TankModel(nCollocation=1,nElements=2,spacing="legendre",bounds=[0,1],params=params)
#Construct snapshots from SVD decomp
U=np.array([[ 1/3, 2/3, -2/3],
            [-2/3, 2/3,  1/3],
            [ 2/3, 1/3,  2/3]
            ])
Ux=np.array([[2,2,0],
             [1,2,0],
             [0,0,1]])
Uxx=np.array([[1,2,0],
              [1,1,0],
              [0,1,2]])
V=np.array([[1,  0, 0],
            [0, -1, 0],
            [0,  0, 1]
            ])
S=np.array([2,1,.1])
A  =np.matmul(U,np.matmul(np.diag(S),V))
Ax =np.matmul(Ux,np.matmul(np.diag(S),V))
Axx=np.matmul(Uxx,np.matmul(np.diag(S),V))
# Compute SVD by QR and Graham-Schmidt
podModes,podModesWeighted, podModesx, podModesxx, timeModes = model.computePODmodes(np.eye(U.shape[0]),A,Ax,Axx,.9)
# Have to do absolute value because SVD non-unique up to sign switches of orthogonal matrices
assert(np.isclose(podModesWeighted.transpose()@podModes,np.eye(podModes.shape[1])).all())
assert(np.isclose(np.abs(podModes[:,:2]),np.abs(U[:,:2])).all())
assert(np.isclose(np.abs(podModesx[:,:2]),np.abs(Ux[:,:2])).all())
assert(np.isclose(np.abs(podModesxx[:,:2]),np.abs(Uxx[:,:2])).all())


#Test POD-ROM matrix calculation
podModes=np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]]).transpose()
podModesx=np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1]]).transpose()
podModesxx=np.array([[0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1],
                     [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]]).transpose()
x=np.linspace(0,1,9)
mean=np.zeros(x.shape)
podModesWeighted, romMassMean, romFirstOrderMat, romFirstOrderMean, romSecondOrderMat, romSecondOrderMean= model.computeRomMatrices(np.eye(x.shape[0]),mean,podModes,podModesx,podModesxx)
podModesWeightedTrue=podModes
romFirstOrderMatTrue=np.array([[1,1],
                           [1/2,1/2]])
romSecondOrderMatTrue=np.array([[1/2,1/2],
                            [1/3,  1/3]])
assert(np.isclose(podModesWeighted,podModesWeightedTrue).all())
# assert(np.isclose(romFirstOrderMat,romFirstOrderMatTrue).all())
# assert(np.isclose(romSecondOrderMat,romSecondOrderMatTrue).all())

#Test Additional Case with Simpsons quad
x=np.linspace(0,1,7)
podModes = np.array([x*0+1,x]).transpose()
mean=x**2
podModesx = np.array([x*0,x*0+1]).transpose()
podModesxx = np.array([x*0,x*0]).transpose()
W=np.diag(np.array([1, 4, 2, 4, 2, 4, 1]))/18
podModesWeighted, romMassMean, romFirstOrderMat, romFirstOrderMean, romSecondOrderMat, romSecondOrderMean= model.computeRomMatrices(W, mean,podModes,podModesx,podModesxx)
assert(np.isclose(podModesWeighted,W@podModes).all())
assert(np.isclose(romMassMean,np.array([1/3,1/4])).all())
assert(np.isclose(podModesWeighted.transpose()@podModes,np.array([[1,1/2],[1/2,1/3]])).all())




