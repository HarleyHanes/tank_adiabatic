import numpy as np

from tankModel.TankModel import TankModel

model = TankModel(nCollocation=2, nElements=5, spacing="legendre", bounds=[0, 1])

# DEIM only Cases
podBasis = np.array([[1, 2, 3], [3, 2, 1], [1, 1, 1]])
nonLinEval = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
expectedBasis = np.array(
    [
        [-1 / np.sqrt(3), 1 / np.sqrt(2), 1 / np.sqrt(6)],
        [-1 / np.sqrt(3), -1 / np.sqrt(2), 1 / np.sqrt(6)],
        [-1 / np.sqrt(3), 0, -2 / np.sqrt(6)],
    ]
)

# Check DEIM basis calculation
deimBasis1, P1 = model.computeDEIMbasis(nonLinEval, 2)
assert np.isclose(deimBasis1, expectedBasis[:, 0:2]).all()
assert np.isclose(P1, np.array([[0, 0], [0, 1], [1, 0]])).all()

deimBasis2, P2 = model.computeDEIMbasis(nonLinEval, 3)
assert np.isclose(deimBasis2, expectedBasis).all()
assert np.isclose(P2, np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])).all()

# Check DEIM Matrix caclulation
deimProjection = model.computeDEIMmatrices(podBasis, deimBasis1, P1)
assert np.isclose(deimProjection, np.array([[4, 1], [4, 1], [4, 1]])).all()


# Full POD Caclulation
#   Here we're just checking the limit case where the DEIM projection is full rank so it should be equivalent to no DEIM
# Define solution (Traveling waves)
t = np.linspace(0, 1, 20)


def u(x, t):
    return 1 + np.sin(2 * np.pi * (x + t)) / 2


def v(x, t):
    return 1 + np.cos(2 * np.pi * (x + t)) / 2


uCoeff = np.empty((t.size, model.collocationPoints.size))
vCoeff = np.empty((t.size, model.collocationPoints.size))
for it in range(t.size):
    for ix in range(model.nCollocation):
        uCoeff[it, ix] = u(t[it], model.collocationPoints[ix])
        vCoeff[it, ix] = u(t[it], model.collocationPoints[ix])
modelCoeff = np.append(uCoeff, vCoeff, axis=1)
# Case 1: no mean, uniform points
x, W = model.getQuadWeights(3, "uniform")

romData, truncationError = model.constructPodRom(
    modelCoeff, x, W, 1, mean="zero", nDeimPoints="none", useEnergyThreshold=False
)
assert np.isclose(romData.uNonLinProjection, romData.uModesWeighted.transpose()).all()
assert np.isclose(romData.vNonLinProjection, romData.vModesWeighted.transpose()).all()
romData, truncationError = model.constructPodRom(
    modelCoeff, x, W, 2, mean="zero", nDeimPoints=3, useEnergyThreshold=False
)
print(romData.uNonLinProjection @ romData.deimProjection - romData.uModesWeighted.transpose())
assert np.isclose(
    romData.uNonLinProjection @ romData.deimProjection,
    romData.uModesWeighted.transpose(),
    atol=1e-07,
).all()
assert np.isclose(
    romData.vNonLinProjection @ romData.deimProjection,
    romData.vModesWeighted.transpose(),
    atol=1e-07,
).all()

# Case 2: mean, uniform points
x, W = model.getQuadWeights(3, "uniform")

romData, truncationError = model.constructPodRom(
    modelCoeff, x, W, 2, mean="zero", nDeimPoints="none", useEnergyThreshold=False
)
assert np.isclose(romData.uNonLinProjection, romData.uModesWeighted.transpose()).all()
assert np.isclose(romData.vNonLinProjection, romData.vModesWeighted.transpose()).all()
romData, truncationError = model.constructPodRom(
    modelCoeff, x, W, 1, mean="zero", nDeimPoints=3, useEnergyThreshold=False
)
print(romData.uNonLinProjection @ romData.deimProjection - romData.uModesWeighted.transpose())
assert np.isclose(
    romData.uNonLinProjection @ romData.deimProjection,
    romData.uModesWeighted.transpose(),
    atol=1e-07,
).all()
assert np.isclose(
    romData.vNonLinProjection @ romData.deimProjection,
    romData.vModesWeighted.transpose(),
    atol=1e-07,
).all()

# Case 3: mean, simpson points
x, W = model.getQuadWeights(3, "simpson")

romData, truncationError = model.constructPodRom(
    modelCoeff, x, W, 1, mean="mean", nDeimPoints="none", useEnergyThreshold=False
)
assert np.isclose(romData.uNonLinProjection, romData.uModesWeighted.transpose()).all()
assert np.isclose(romData.vNonLinProjection, romData.vModesWeighted.transpose()).all()
romData, truncationError = model.constructPodRom(
    modelCoeff, x, W, 1, mean="mean", nDeimPoints=3, useEnergyThreshold=False
)
assert np.isclose(
    romData.uNonLinProjection @ romData.deimProjection,
    romData.uModesWeighted.transpose(),
    atol=1e-07,
).all()
assert np.isclose(
    romData.vNonLinProjection @ romData.deimProjection,
    romData.vModesWeighted.transpose(),
    atol=1e-07,
).all()
