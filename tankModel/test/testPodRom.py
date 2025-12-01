import numpy as np

from tankModel.romData import RomData
from tankModel.TankModel import TankModel

# Test POD mode Calculation
params = {
    "PeM": 300,
    "PeT": 100,
    "f": 0,
    "Le": 3,
    "Da": 0,
    "beta": 0,
    "gamma": 0,
    "delta": 2,
    "vH": 1,
}
# Define nElem=1,nColl=2 model with roots at -1 and 1 for simplicity
model = TankModel(nCollocation=1, nElements=2, spacing="legendre", bounds=[0, 1], params=params)
# Construct snapshots from SVD decomp
U = np.array([[1 / 3, 2 / 3, -2 / 3], [-2 / 3, 2 / 3, 1 / 3], [2 / 3, 1 / 3, 2 / 3]])
Ux = np.array([[2, 2, 0], [1, 2, 0], [0, 0, 1]])
Uxx = np.array([[1, 2, 0], [1, 1, 0], [0, 1, 2]])
V = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
S = np.array([2, 1, 0.1])
A = np.matmul(U, np.matmul(np.diag(S), V))
Ax = np.matmul(Ux, np.matmul(np.diag(S), V))
Axx = np.matmul(Uxx, np.matmul(np.diag(S), V))
# Compute SVD by QR and Graham-Schmidt
podModes, podModesx, podModesxx, timeModes, podError = model.computePODmodes(np.eye(U.shape[0]), A, Ax, Axx, 0.9)
# Have to do absolute value because SVD non-unique up to sign switches of orthogonal matrices
assert np.isclose(np.abs(podModes[:, :2]), np.abs(U[:, :2])).all()
assert np.isclose(np.abs(podModesx[:, :2]), np.abs(Ux[:, :2])).all()
assert np.isclose(np.abs(podModesxx[:, :2]), np.abs(Uxx[:, :2])).all()


# Test ROM matrix calculation
model = TankModel(nCollocation=1, nElements=2, spacing="legendre", bounds=[0, 1], params=params)
# Exact uniform case
x = np.linspace(0, 1, 9)
podModes = np.array([x * 0 + 1, x]).transpose()
podModesx = np.array([x * 0 + 1, x * 0 + 1]).transpose()
podModesxx = np.array([x, x]).transpose()
mean = np.zeros(x.shape)
meanx = np.zeros(x.shape)
meanxx = np.zeros(x.shape)
W = model.getQuadWeights(x, "uniform")
(
    podModesWeighted,
    podModesInt,
    romMassMean,
    romFirstOrderMat,
    romFirstOrderMean,
    romSecondOrderMat,
    romSecondOrderMean,
) = model.computeRomMatrices(W, mean, meanx, meanxx, podModes, podModesx, podModesxx)
podModesWeightedTrue = podModes
romFirstOrderMatTrue = np.array([[1, 1], [1 / 2, 1 / 2]])
romSecondOrderMatTrue = np.array([[1 / 2, 1 / 2], [1 / 3, 1 / 3]])
assert np.isclose(podModesWeighted, podModesWeightedTrue).all()
# assert(np.isclose(romFirstOrderMat,romFirstOrderMatTrue).all())
# assert(np.isclose(romSecondOrderMat,romSecondOrderMatTrue).all())

# Exact simpson case
x = np.linspace(0, 1, 7)
mean = x**2
meanx = 2 * x
meanxx = 2 + 0 * x
podModes = np.array([x * 0 + 1, x]).transpose()
podModesx = np.array([x * 0, x * 0 + 1]).transpose()
podModesxx = np.array([x * 0, x * 0]).transpose()
W = model.getQuadWeights(x, "simpson")
(
    podModesWeighted,
    podModesInt,
    romMassMean,
    romFirstOrderMat,
    romFirstOrderMean,
    romSecondOrderMat,
    romSecondOrderMean,
) = model.computeRomMatrices(W, mean, meanx, meanxx, podModes, podModesx, podModesxx)
assert np.isclose(podModesWeighted, W @ podModes).all()
assert np.isclose(romMassMean, np.array([1 / 3, 1 / 4])).all()
assert np.isclose(podModesWeighted.transpose() @ podModes, np.array([[1, 1 / 2], [1 / 2, 1 / 3]])).all()
# assert(np.isclose(romFirstOrderMat,).all())
# assert(np.isclose(romFirstOrderMean,).all())
# assert(np.isclose(romSecondOrderMean,).all())
# assert(np.isclose(romSecondOrderMat,).all())

# #Inexact Simpson case
# x=np.linspace(0,1,100)
# mean = x**2
# meanx = 2*x
# meanxx = 2+0*x
# podModes = np.array([x,x**2,x**3]).transpose()
# Inexact Simpson case (optional)
# podModesx = np.array([x**3, x**2, x]).transpose()
# podModesxx = np.array([x**2, x**2, x**2]).transpose()
# podModesWeighted, romMassMean, romFirstOrderMat, romFirstOrderMean,
# romSecondOrderMat, romSecondOrderMean = model.computeRomMatrices(
#     W, mean, meanx, meanxx, podModes, podModesx, podModesxx
# )


# Legendre Modes case - Continuous
model = TankModel(nCollocation=1, nElements=2, spacing="legendre", bounds=[-1, 1], params=params)
# Get points and quad weights matrices
x = np.linspace(-1, 1, 11)
W = model.getQuadWeights(x, "simpson")

uModes = np.array([1 + x * 0, x]).transpose()
uModesx = np.array([x * 0, 1 + x * 0]).transpose()
uModesxx = np.array([x * 0, x * 0]).transpose()

vModes = np.array([1 + x * 0, 1 / 2 * (3 * x**2 - 1)]).transpose()
vModesx = np.array([x * 0, 3 * x]).transpose()
vModesxx = np.array([x * 0, 3 + x * 0]).transpose()

mean = 1 + x
meanx = 1 + x * 0
meanxx = x * 0

(
    uModesWeighted,
    uModesInt,
    uMassMean,
    uFirstOrderMat,
    uFirstOrderMean,
    uSecondOrderMat,
    uSecondOrderMean,
) = model.computeRomMatrices(W, mean, meanx, meanxx, uModes, uModesx, uModesxx)
(
    vModesWeighted,
    vModesInt,
    vMassMean,
    vFirstOrderMat,
    vFirstOrderMean,
    vSecondOrderMat,
    vSecondOrderMean,
) = model.computeRomMatrices(W, mean, meanx, meanxx, vModes, vModesx, vModesxx)


assert np.isclose(uMassMean, np.array([2, 2 / 3])).all()
assert np.isclose(uFirstOrderMat, np.array([[0, 2], [0, 0]])).all()
assert np.isclose(uFirstOrderMean, np.array([2, 0])).all()
assert np.isclose(uSecondOrderMat, np.array([[0, 0], [0, 0]])).all()
assert np.isclose(uSecondOrderMean, np.array([0, 0])).all()

assert np.isclose(vMassMean, np.array([2, 0])).all()
assert np.isclose(vFirstOrderMat, np.array([[0, 0], [0, 0]])).all()
assert np.isclose(vFirstOrderMean, np.array([2, 0])).all()
assert np.isclose(vSecondOrderMat, np.array([[0, 6], [0, 0]])).all()
assert np.isclose(vSecondOrderMean, np.array([0, 0])).all()

# Test dydt evaluations
# Save modes and rom matrices from legendre case to model
romData = RomData(
    x,
    W,
    np.zeros(uModes.shape),
    mean,
    uModes,
    uModesx,
    uModesxx,
    uModesWeighted,
    uModesInt,
    uMassMean,
    uFirstOrderMat,
    uFirstOrderMean,
    uSecondOrderMat,
    uSecondOrderMean,
    np.zeros(uModes.shape),
    mean,
    vModes,
    vModesx,
    vModesxx,
    vModesWeighted,
    vModesInt,
    vMassMean,
    vFirstOrderMat,
    vFirstOrderMean,
    vSecondOrderMat,
    vSecondOrderMean,
)
# Case 1: Linear cases
params = {
    "PeM": 100,
    "PeT": 10,
    "f": 0,
    "Le": 3,
    "Da": 0,
    "beta": 0,
    "gamma": 0,
    "delta": 1.1,
    "vH": 2,
}
model.params = params
y = np.array([1, 0, 0, 0])
dydt = model.dydtPodRom(y, 0, romData)
assert np.isclose(
    dydt[0:2],
    1 / params["PeM"] * uSecondOrderMat[:, 0]
    - uFirstOrderMat[:, 0]
    + 1 / params["PeM"] * uSecondOrderMean
    - uFirstOrderMean,
).all()
assert (
    np.isclose(
        dydt[2:4],
        (
            1 / params["PeT"] * vSecondOrderMean
            - vFirstOrderMean
            + params["delta"] * (params["vH"] * vModesInt - vMassMean)
        )
        / params["Le"],
    )
).all()

y = np.array([0, 1, 0, 0])
dydt = model.dydtPodRom(y, 0, romData)
assert np.isclose(
    dydt[0:2],
    1 / params["PeM"] * uSecondOrderMat[:, 1]
    - uFirstOrderMat[:, 1]
    + 1 / params["PeM"] * uSecondOrderMean
    - uFirstOrderMean,
).all()
assert (
    np.isclose(
        dydt[2:4],
        (
            1 / params["PeT"] * vSecondOrderMean
            - vFirstOrderMean
            + params["delta"] * (params["vH"] * vModesInt - vMassMean)
        )
        / params["Le"],
    )
).all()


y = np.array([0, 0, 1, 0])
dydt = model.dydtPodRom(y, 0, romData)
assert np.isclose(dydt[0:2], 1 / params["PeM"] * uSecondOrderMean - uFirstOrderMean).all()
assert (
    np.isclose(
        dydt[2:4],
        (
            1 / params["PeT"] * vSecondOrderMat[:, 0]
            - vFirstOrderMat[:, 0]
            - params["delta"] * np.array([1, 0])
            + 1 / params["PeT"] * vSecondOrderMean
            - vFirstOrderMean
            + params["delta"] * (params["vH"] * vModesInt - vMassMean)
        )
        / params["Le"],
    )
).all()

y = np.array([0, 0, 0, 1])
dydt = model.dydtPodRom(y, 0, romData)
assert np.isclose(dydt[0:2], 1 / params["PeM"] * uSecondOrderMean - uFirstOrderMean).all()
assert (
    np.isclose(
        dydt[2:4],
        (
            1 / params["PeT"] * vSecondOrderMat[:, 1]
            - vFirstOrderMat[:, 1]
            - params["delta"] * np.array([0, 1])
            + 1 / params["PeT"] * vSecondOrderMean
            - vFirstOrderMean
            + params["delta"] * (params["vH"] * vModesInt - vMassMean)
        )
        / params["Le"],
    )
).all()

# Case 2: Nonlinear cases
params = {"PeM": 1, "PeT": 1, "f": 0, "Le": 1, "Da": 1, "beta": 1, "gamma": 1, "delta": 0, "vH": 0}
model.params = params
y = np.array([1, 0, 0, 0])
dydt = model.dydtPodRom(y, 0, romData)
assert np.isclose(
    dydt[0:2],
    1 / params["PeM"] * uSecondOrderMat[:, 0]
    - uFirstOrderMat[:, 0]
    + 1 / params["PeM"] * uSecondOrderMean
    - uFirstOrderMean
    + params["Da"]
    * uModesWeighted.transpose()
    @ ((1 - (uModes[:, 0] + mean)) * np.exp(params["beta"] * params["gamma"] * mean / (1 + params["beta"] * mean))),
).all()
assert (
    np.isclose(
        dydt[2:4],
        (
            1 / params["PeT"] * vSecondOrderMean
            - vFirstOrderMean
            + params["delta"] * (params["vH"] * vModesInt - vMassMean)
            + params["Da"]
            * vModesWeighted.transpose()
            @ (
                (1 - (uModes[:, 0] + mean))
                * np.exp(params["beta"] * params["gamma"] * mean / (1 + params["beta"] * mean))
            )
        )
        / params["Le"],
    )
).all()

y = np.array([0, 1, 0, 0])
dydt = model.dydtPodRom(y, 0, romData)
assert np.isclose(
    dydt[0:2],
    1 / params["PeM"] * uSecondOrderMat[:, 1]
    - uFirstOrderMat[:, 1]
    + 1 / params["PeM"] * uSecondOrderMean
    - uFirstOrderMean
    + params["Da"]
    * uModesWeighted.transpose()
    @ ((1 - (uModes[:, 1] + mean)) * np.exp(params["beta"] * params["gamma"] * mean / (1 + params["beta"] * mean))),
).all()
assert (
    np.isclose(
        dydt[2:4],
        (
            1 / params["PeT"] * vSecondOrderMean
            - vFirstOrderMean
            + params["delta"] * (params["vH"] * vModesInt - vMassMean)
            + params["Da"]
            * vModesWeighted.transpose()
            @ (
                (1 - (uModes[:, 1] + mean))
                * np.exp(params["beta"] * params["gamma"] * mean / (1 + params["beta"] * mean))
            )
        )
        / params["Le"],
    )
).all()

y = np.array([0, 0, 1, 0])
dydt = model.dydtPodRom(y, 0, romData)
assert np.isclose(
    dydt[0:2],
    1 / params["PeM"] * uSecondOrderMean
    - uFirstOrderMean
    + params["Da"]
    * uModesWeighted.transpose()
    @ (
        (1 - mean)
        * np.exp(
            params["beta"] * params["gamma"] * (vModes[:, 0] + mean) / (1 + params["beta"] * (vModes[:, 0] + mean))
        )
    ),
).all()
assert (
    np.isclose(
        dydt[2:4],
        (
            1 / params["PeT"] * vSecondOrderMat[:, 0]
            - vFirstOrderMat[:, 0]
            - params["delta"] * np.array([1, 0])
            + 1 / params["PeT"] * vSecondOrderMean
            - vFirstOrderMean
            + params["delta"] * (params["vH"] * vModesInt - vMassMean)
            + params["Da"]
            * vModesWeighted.transpose()
            @ (
                (1 - mean)
                * np.exp(
                    params["beta"]
                    * params["gamma"]
                    * (vModes[:, 0] + mean)
                    / (1 + params["beta"] * (vModes[:, 0] + mean))
                )
            )
        )
        / params["Le"],
    )
).all()

y = np.array([0, 0, 0, 1])
dydt = model.dydtPodRom(y, 0, romData)
assert np.isclose(
    dydt[0:2],
    1 / params["PeM"] * uSecondOrderMean
    - uFirstOrderMean
    + params["Da"]
    * uModesWeighted.transpose()
    @ (
        (1 - mean)
        * np.exp(
            params["beta"] * params["gamma"] * (vModes[:, 1] + mean) / (1 + params["beta"] * (vModes[:, 1] + mean))
        )
    ),
).all()
assert (
    np.isclose(
        dydt[2:4],
        (
            1 / params["PeT"] * vSecondOrderMat[:, 1]
            - vFirstOrderMat[:, 1]
            - params["delta"] * np.array([0, 1])
            + 1 / params["PeT"] * vSecondOrderMean
            - vFirstOrderMean
            + params["delta"] * (params["vH"] * vModesInt - vMassMean)
            + params["Da"]
            * vModesWeighted.transpose()
            @ (
                (1 - mean)
                * np.exp(
                    params["beta"]
                    * params["gamma"]
                    * (vModes[:, 1] + mean)
                    / (1 + params["beta"] * (vModes[:, 1] + mean))
                )
            )
        )
        / params["Le"],
    )
).all()

# Legendre Modes case - Discretized
params = {"PeM": 1, "PeT": 1, "f": 0, "Le": 3, "Da": 0, "beta": 0, "gamma": 0, "delta": 2, "vH": 1}
model = TankModel(nCollocation=2, nElements=2, spacing="legendre", bounds=[-1, 1], params=params)
# Get points and quad weights matrices
x = np.linspace(-1, 1, 11)
# Construct model Coeff
# NEED TO UPDATE MODES SO SATISFY BC
modelCoeff = np.array(
    [
        np.append(1 + model.collocationPoints * 0, 1 + model.collocationPoints * 0),
        np.append(model.collocationPoints, 1 / 2 * (3 * model.collocationPoints**2 - 1)),
    ]
)

uModes = np.array([1 / 2 + x * 0, x]).transpose()
uModesx = np.array([x * 0, 1 + x * 0]).transpose()
uModesxx = np.array([x * 0, x * 0]).transpose()

vModes = np.array([1 + x * 0, 1 / 2 * (3 * x**2 - 1)]).transpose()
vModesx = np.array([x * 0, 3 * x]).transpose()
vModesxx = np.array([x * 0, 3 + x * 0]).transpose()

# mean   = 1+x
# meanx   = 1+x*0
# meanxx   = x*0
# uMean = 1+x-x
# vMean = 1+x-x
# mean = np.append(uMean,vMean)
romData, truncationError = model.constructPodRom(
    modelCoeff, x, 2, quadRule="simpson", mean="zero", useEnergyThreshold=False
)

print(romData.uModes)
assert np.isclose(model.eval(x, modelCoeff, output="u").transpose(), uModes).all()
assert np.isclose(model.eval(x, modelCoeff, output="v").transpose(), vModes).all()
assert np.isclose(romData.uModes, uModes).all()
assert np.isclose(romData.vModes, vModes).all()
# assert(np.isclose(romData.uMassMean,np.array([2,2/3])).all())
assert np.isclose(romData.uRomFirstOrderMat, np.array([[0, 2], [0, 0]])).all()
assert np.isclose(romData.uRomFirstOrderMean, np.array([2, 0])).all()
assert np.isclose(romData.uRomSecondOrderMat, np.array([[0, 0], [0, 0]])).all()
assert np.isclose(romData.uRomSecondOrderMean, np.array([0, 0])).all()

# assert(np.isclose(vMassMean,np.array([2,0])).all())
assert np.isclose(romData.vRomFirstOrderMat, np.array([[0, 0], [0, 0]])).all()
assert np.isclose(romData.vRomFirstOrderMean, np.array([2, 0])).all()
assert np.isclose(romData.vRomSecondOrderMat, np.array([[0, 6], [0, 0]])).all()
assert np.isclose(romData.vRomSecondOrderMean, np.array([0, 0])).all()

# Test dydt evaluations

# Case 1: Linear cases
params = {
    "PeM": 100,
    "PeT": 10,
    "f": 0,
    "Le": 3,
    "Da": 0,
    "beta": 0,
    "gamma": 0,
    "delta": 1.1,
    "vH": 2,
}
model.params = params
y = np.array([1, 0, 0, 0])
dydt = model.dydtPodRom(y, 0, romData)
assert np.isclose(
    dydt[0:2],
    1 / params["PeM"] * uSecondOrderMat[:, 0]
    - uFirstOrderMat[:, 0]
    + 1 / params["PeM"] * uSecondOrderMean
    - uFirstOrderMean,
).all()
assert (
    np.isclose(
        dydt[2:4],
        (
            1 / params["PeT"] * vSecondOrderMean
            - vFirstOrderMean
            + params["delta"] * (params["vH"] * vModesInt - vMassMean)
        )
        / params["Le"],
    )
).all()

y = np.array([0, 1, 0, 0])
dydt = model.dydtPodRom(y, 0, romData)
assert np.isclose(
    dydt[0:2],
    1 / params["PeM"] * uSecondOrderMat[:, 1]
    - uFirstOrderMat[:, 1]
    + 1 / params["PeM"] * uSecondOrderMean
    - uFirstOrderMean,
).all()
assert (
    np.isclose(
        dydt[2:4],
        (
            1 / params["PeT"] * vSecondOrderMean
            - vFirstOrderMean
            + params["delta"] * (params["vH"] * vModesInt - vMassMean)
        )
        / params["Le"],
    )
).all()


y = np.array([0, 0, 1, 0])
dydt = model.dydtPodRom(y, 0, romData)
assert np.isclose(dydt[0:2], 1 / params["PeM"] * uSecondOrderMean - uFirstOrderMean).all()
assert (
    np.isclose(
        dydt[2:4],
        (
            1 / params["PeT"] * vSecondOrderMat[:, 0]
            - vFirstOrderMat[:, 0]
            - params["delta"] * np.array([1, 0])
            + 1 / params["PeT"] * vSecondOrderMean
            - vFirstOrderMean
            + params["delta"] * (params["vH"] * vModesInt - vMassMean)
        )
        / params["Le"],
    )
).all()

y = np.array([0, 0, 0, 1])
dydt = model.dydtPodRom(y, 0, romData)
assert np.isclose(dydt[0:2], 1 / params["PeM"] * uSecondOrderMean - uFirstOrderMean).all()
assert (
    np.isclose(
        dydt[2:4],
        (
            1 / params["PeT"] * vSecondOrderMat[:, 1]
            - vFirstOrderMat[:, 1]
            - params["delta"] * np.array([0, 1])
            + 1 / params["PeT"] * vSecondOrderMean
            - vFirstOrderMean
            + params["delta"] * (params["vH"] * vModesInt - vMassMean)
        )
        / params["Le"],
    )
).all()

# Case 2: Nonlinear cases
params = {"PeM": 1, "PeT": 1, "f": 0, "Le": 1, "Da": 1, "beta": 1, "gamma": 1, "delta": 0, "vH": 0}
model.params = params
y = np.array([1, 0, 0, 0])
dydt = model.dydtPodRom(y, 0, romData)
assert np.isclose(
    dydt[0:2],
    1 / params["PeM"] * uSecondOrderMat[:, 0]
    - uFirstOrderMat[:, 0]
    + 1 / params["PeM"] * uSecondOrderMean
    - uFirstOrderMean
    + params["Da"]
    * uModesWeighted.transpose()
    @ ((1 - (uModes[:, 0] + mean)) * np.exp(params["beta"] * params["gamma"] * mean / (1 + params["beta"] * mean))),
).all()
assert (
    np.isclose(
        dydt[2:4],
        (
            1 / params["PeT"] * vSecondOrderMean
            - vFirstOrderMean
            + params["delta"] * (params["vH"] * vModesInt - vMassMean)
            + params["Da"]
            * vModesWeighted.transpose()
            @ (
                (1 - (uModes[:, 0] + mean))
                * np.exp(params["beta"] * params["gamma"] * mean / (1 + params["beta"] * mean))
            )
        )
        / params["Le"],
    )
).all()

y = np.array([0, 1, 0, 0])
dydt = model.dydtPodRom(y, 0, romData)
assert np.isclose(
    dydt[0:2],
    1 / params["PeM"] * uSecondOrderMat[:, 1]
    - uFirstOrderMat[:, 1]
    + 1 / params["PeM"] * uSecondOrderMean
    - uFirstOrderMean
    + params["Da"]
    * uModesWeighted.transpose()
    @ ((1 - (uModes[:, 1] + mean)) * np.exp(params["beta"] * params["gamma"] * mean / (1 + params["beta"] * mean))),
).all()
assert (
    np.isclose(
        dydt[2:4],
        (
            1 / params["PeT"] * vSecondOrderMean
            - vFirstOrderMean
            + params["delta"] * (params["vH"] * vModesInt - vMassMean)
            + params["Da"]
            * vModesWeighted.transpose()
            @ (
                (1 - (uModes[:, 1] + mean))
                * np.exp(params["beta"] * params["gamma"] * mean / (1 + params["beta"] * mean))
            )
        )
        / params["Le"],
    )
).all()

y = np.array([0, 0, 1, 0])
dydt = model.dydtPodRom(y, 0, romData)
assert np.isclose(
    dydt[0:2],
    1 / params["PeM"] * uSecondOrderMean
    - uFirstOrderMean
    + params["Da"]
    * uModesWeighted.transpose()
    @ (
        (1 - mean)
        * np.exp(
            params["beta"] * params["gamma"] * (vModes[:, 0] + mean) / (1 + params["beta"] * (vModes[:, 0] + mean))
        )
    ),
).all()
assert (
    np.isclose(
        dydt[2:4],
        (
            1 / params["PeT"] * vSecondOrderMat[:, 0]
            - vFirstOrderMat[:, 0]
            - params["delta"] * np.array([1, 0])
            + 1 / params["PeT"] * vSecondOrderMean
            - vFirstOrderMean
            + params["delta"] * (params["vH"] * vModesInt - vMassMean)
            + params["Da"]
            * vModesWeighted.transpose()
            @ (
                (1 - mean)
                * np.exp(
                    params["beta"]
                    * params["gamma"]
                    * (vModes[:, 0] + mean)
                    / (1 + params["beta"] * (vModes[:, 0] + mean))
                )
            )
        )
        / params["Le"],
    )
).all()

y = np.array([0, 0, 0, 1])
dydt = model.dydtPodRom(y, 0, romData)
assert np.isclose(
    dydt[0:2],
    1 / params["PeM"] * uSecondOrderMean
    - uFirstOrderMean
    + params["Da"]
    * uModesWeighted.transpose()
    @ (
        (1 - mean)
        * np.exp(
            params["beta"] * params["gamma"] * (vModes[:, 1] + mean) / (1 + params["beta"] * (vModes[:, 1] + mean))
        )
    ),
).all()
assert (
    np.isclose(
        dydt[2:4],
        (
            1 / params["PeT"] * vSecondOrderMat[:, 1]
            - vFirstOrderMat[:, 1]
            - params["delta"] * np.array([0, 1])
            + 1 / params["PeT"] * vSecondOrderMean
            - vFirstOrderMean
            + params["delta"] * (params["vH"] * vModesInt - vMassMean)
            + params["Da"]
            * vModesWeighted.transpose()
            @ (
                (1 - mean)
                * np.exp(
                    params["beta"]
                    * params["gamma"]
                    * (vModes[:, 1] + mean)
                    / (1 + params["beta"] * (vModes[:, 1] + mean))
                )
            )
        )
        / params["Le"],
    )
).all()
