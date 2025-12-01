import matplotlib.pyplot as plt
import numpy as np

from tankModel.TankModel import TankModel

nCollocation = 1
nElements = 2
params = {"PeM": 1, "PeT": 1, "f": 1, "Le": 0, "Da": 0, "beta": 0, "gamma": 0, "delta": 0, "vH": 0}
model = TankModel(
    nCollocation=nCollocation,
    nElements=nElements,
    spacing="legendre",
    bounds=[-2, 2],
    params=params,
)
x = np.linspace(-2, 2, 100)

plt.plot(x, model.elements[0].basisFunctions(x).transpose())
plt.show()
plt.plot(x, model.elements[0].basisFirstDeriv(x).transpose())
plt.show()
