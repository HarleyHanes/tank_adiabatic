import sys
import os 
current_script_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath(os.path.join(current_script_dir,'..','..'))
sys.path.append(grandparent_dir)
print(sys.path)

import numpy as np
from tankModel.TankModel import TankModel
from tankModel.romData import RomData
import scipy
model = TankModel(nCollocation=1,nElements=2,spacing="legendre",bounds=[0,1])

podBasis = np.array([[1,2,3],[3,2,1],[1,1,1]])
nonLinEval = np.array([[1, 1, 0],[1,0,1],[0,1,1]])
expectedBasis = np.array([[-1/np.sqrt(3), 1/np.sqrt(6),1/np.sqrt(2)],[-1/np.sqrt(3),1/np.sqrt(6),-1/np.sqrt(2)],[-1/np.sqrt(3),-2/np.sqrt(6),0]])

#Check DEIM basis calculation
deimBasis1,P1 = model.computeDEIMbasis(nonLinEval,2)
assert(np.isclose(deimBasis1, expectedBasis[:,0:2]).all())
assert(np.isclose(P1, np.array([[1,0],[0,0],[0,1]])).all())

deimBasis2,P2 = model.computeDEIMbasis(nonLinEval,3)
assert(np.isclose(deimBasis2, expectedBasis).all())
assert(np.isclose(P2, np.array([[1,0,0],[0,0,1],[0,1,0]])).all())

#Check DEIM Matrix caclulation
deimProjection, podProjection = model.computeDEIMmatrices(podBasis,deimBasis1,P1)
print(deimProjection)
assert(np.isclose(deimProjection,np.array([[4,1],[4,1],[4,1]])).all())
assert(np.isclose(podProjection,np.array([[1,2,3],[1,1,1]])).all())

