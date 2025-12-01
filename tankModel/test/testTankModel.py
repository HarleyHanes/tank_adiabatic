import numpy as np

from tankModel.TankModel import TankModel

print("Running testTankModel.py")
print("     Testing Computation Tools")
# Check Eval
nCollocation = 2
nElements = 2
params = {"PeM": 10, "PeT": 1, "f": 0, "Le": 1, "Da": 0, "beta": 0, "gamma": 0, "delta": 0, "vH": 0}
model = TankModel(nCollocation=nCollocation, nElements=nElements, spacing="legendre", bounds=[0, 1], params=params)


# Have to use f that satisfied boundary conditions
def f(x):
    return -(x**2) + 2 * x + 2 / 10


def fx(x):
    return -2 * x + 2


def fxx(x):
    return -2 + x * 0


modelCoeff = f(np.append(model.collocationPoints, model.collocationPoints, axis=0))
xEval = np.linspace(0, 1, 10)
fEval = model.eval(xEval, modelCoeff, output="u")
fxEval = model.eval(xEval, modelCoeff, output="u", deriv=1)
fxxEval = model.eval(xEval, modelCoeff, output="u", deriv=2)
assert np.isclose(fEval, f(xEval)).all()
assert np.isclose(fxEval, fx(xEval)).all()
assert np.isclose(fxxEval, fxx(xEval)).all()
print("         Model Evaluation Passing")


# Check Integration: Spatial Integration Only
nCollocation = 2
nElements = 2
model = TankModel(nCollocation=nCollocation, nElements=nElements, spacing="legendre", bounds=[-2, 2])


def f(x):
    return x**2 + x + 1


fint = 9 + 1 / 3
integral = model.integrateSpace(f)

nCollocation = 3
nElements = 2
model = TankModel(nCollocation=nCollocation, nElements=nElements, spacing="legendre", bounds=[0, 2])


def f(x):
    return x**3 + x**2 + x + 1


fint = 10 + 2 / 3
integral = model.integrateSpace(f)
assert np.isclose(fint, integral)
# Check Integration: Spatial at multiple points and temporal
nCollocation = 2
nElements = 2
model = TankModel(nCollocation=nCollocation, nElements=nElements, spacing="legendre", bounds=[-2, 2])


def f(x):
    return np.outer(np.array([0, 1, 2]), x**2 + x + 1).flatten()


fint = np.array([0, 9 + 1 / 3, 18 + 2 / 3])
fTempint = 18 + 2 / 3
integral, integralSpace = model.integrate(f, np.array([0, 1, 2]))
assert np.isclose(fTempint, integral)
assert np.isclose(fint, integralSpace).all()
print("         Integration Passsing")
print("     Testing Model Matrices")
params = {"PeM": 1, "PeT": 1, "f": 1, "Le": 1, "Da": 1, "beta": 1, "gamma": 1, "delta": 1, "vH": 1}
model = TankModel(nCollocation=1, nElements=2, spacing="legendre", bounds=[0, 1], params=params)
x = np.array([0, 1, 2, 3, 4]) / 4
l10 = 8 * x**2 - 6 * x + 1
l10x = 16 * x - 6
l11 = 8 * x - 16 * x**2
l11x = 8 - 32 * x
l12 = 8 * x**2 - 2 * x
l12x = 16 * x - 2
l20 = 8 * x**2 - 14 * x + 6
l20x = 16 * x - 14
l21 = -16 * x**2 + 24 * x - 8
l21x = -32 * x + 24
l22 = 8 * x**2 - 10 * x + 3
l22x = 16 * x - 10
trueMassBoundaryMat = np.array(
    [
        [l10x[0] - params["PeM"], l11x[0], l12x[0], 0, 0],
        [0, 1, 0, 0, 0],
        [l10x[2], l11x[2], l12x[2] - l20x[2], -l21x[2], -l22x[2]],
        [0, 0, 0, 1, 0],
        [0, 0, l20x[4], l21x[4], l22x[4]],
    ]
)
trueTempBoundaryMat = np.array(
    [
        [l10x[0] - params["PeT"], l11x[0], l12x[0], 0, params["f"] * params["PeT"]],
        [0, 1, 0, 0, 0],
        [l10x[2], l11x[2], l12x[2] - l20x[2], -l21x[2], -l22x[2]],
        [0, 0, 0, 1, 0],
        [0, 0, l20x[4], l21x[4], l22x[4]],
    ]
)
# Check Boundary matrices
assert np.isclose(trueMassBoundaryMat, model.massBoundaryMat).all()
assert np.isclose(trueTempBoundaryMat, model.tempBoundaryMat).all()

print("         Boundary Matrices Passing")

model = TankModel(nCollocation=2, nElements=2, spacing="uniform", bounds=[-3, 3], params=params)
x = model.collocationPoints
l10 = -1 / 6 * (x**3 + 3 * x**2 + 2 * x)
l10x = -1 / 6 * (3 * x**2 + 6 * x + 2)
l10xx = -(x + 1)
l11 = 1 / 2 * (x**3 + 4 * x**2 + 3 * x)
l11x = 1 / 2 * (3 * x**2 + 8 * x + 3)
l11xx = 3 * x + 4
l12 = -1 / 2 * (x**3 + 5 * x**2 + 6 * x)
l12x = -1 / 2 * (3 * x**2 + 10 * x + 6)
l12xx = -3 * x - 5
l13 = 1 / 6 * (x**3 + 6 * x**2 + 11 * x + 6)
l13x = 1 / 6 * (3 * x**2 + 12 * x + 11)
l13xx = x + 2
l20 = -1 / 6 * (x**3 - 6 * x**2 + 11 * x - 6)
l20x = -1 / 6 * (3 * x**2 - 12 * x + 11)
l20xx = 2 - x
l21 = 1 / 2 * (x**3 - 5 * x**2 + 6 * x)
l21x = 1 / 2 * (3 * x**2 - 10 * x + 6)
l21xx = 3 * x - 5
l22 = -1 / 2 * (x**3 - 4 * x**2 + 3 * x)
l22x = -1 / 2 * (3 * x**2 - 8 * x + 3)
l22xx = 4 - 3 * x
l23 = 1 / 6 * (x**3 - 3 * x**2 + 2 * x)
l23x = 1 / 6 * (3 * x**2 - 6 * x + 2)
l23xx = x - 1

#
expectedFirstOrderMat = np.zeros((4, 7))
expectedFirstOrderMat[0, 0:4] = np.array([l10x[0], l11x[0], l12x[0], l13x[0]])
expectedFirstOrderMat[1, 0:4] = np.array([l10x[1], l11x[1], l12x[1], l13x[1]])
expectedFirstOrderMat[2, 3:7] = np.array([l20x[2], l21x[2], l22x[2], l23x[2]])
expectedFirstOrderMat[3, 3:7] = np.array([l20x[3], l21x[3], l22x[3], l23x[3]])
assert np.isclose(expectedFirstOrderMat, model.firstOrderMat).all()
expectedSecondOrderMat = np.zeros((4, 7))
expectedSecondOrderMat[0, 0:4] = np.array([l10xx[0], l11xx[0], l12xx[0], l13xx[0]])
expectedSecondOrderMat[1, 0:4] = np.array([l10xx[1], l11xx[1], l12xx[1], l13xx[1]])
expectedSecondOrderMat[2, 3:7] = np.array([l20xx[2], l21xx[2], l22xx[2], l23xx[2]])
expectedSecondOrderMat[3, 3:7] = np.array([l20xx[3], l21xx[3], l22xx[3], l23xx[3]])
assert np.isclose(expectedSecondOrderMat, model.secondOrderMat).all()
print("         Advec/ Diffusion Matrices Passing")

params = {
    "PeM": 10,
    "PeT": 12,
    "f": 1,
    "Le": 1,
    "Da": 1,
    "beta": 1,
    "gamma": 1,
    "delta": 1,
    "vH": 1,
}
model = TankModel(nCollocation=1, nElements=2, spacing="legendre", bounds=[0, 1], params=params)
x = np.array([0, 1, 2, 3, 4]) / 4
l10 = 8 * x**2 - 6 * x + 1
l10x = 16 * x - 6
l11 = 8 * x - 16 * x**2
l11x = 8 - 32 * x
l12 = 8 * x**2 - 2 * x
l12x = 16 * x - 2
l20 = 8 * x**2 - 14 * x + 6
l20x = 16 * x - 14
l21 = -16 * x**2 + 24 * x - 8
l21x = -32 * x + 24
l22 = 8 * x**2 - 10 * x + 3
l22x = 16 * x - 10
peMsensBoundaryMat = np.zeros(
    (
        2 * ((model.nCollocation + 1) * model.nElements + 1),
        2 * (model.nCollocation + 1) * model.nElements + 2,
    )
)
peMsensBoundaryMat[0:5, 0:5] = np.array(
    [
        [l10x[0] - params["PeM"], l11x[0], l12x[0], 0, 0],
        [0, 1, 0, 0, 0],
        [l10x[2], l11x[2], l12x[2] - l20x[2], -l21x[2], -l22x[2]],
        [0, 0, 0, 1, 0],
        [0, 0, l20x[4], l21x[4], l22x[4]],
    ]
)
peMsensBoundaryMat[5:, 5:] = peMsensBoundaryMat[0:5, 0:5]
peMsensBoundaryMat[5, 0] = -1
# Check Boundary matrices
assert np.isclose(model.dudPeMboundaryMat, peMsensBoundaryMat).all()
print("         PeM Sensitivity Boundary Matrix Passing")

params = {
    "PeM": 10,
    "PeT": 12,
    "f": 2,
    "Le": 1,
    "Da": 1,
    "beta": 1,
    "gamma": 1,
    "delta": 1,
    "vH": 1,
}
model = TankModel(nCollocation=1, nElements=2, spacing="legendre", bounds=[0, 1], params=params)
x = np.array([0, 1, 2, 3, 4]) / 4
l10 = 8 * x**2 - 6 * x + 1
l10x = 16 * x - 6
l11 = 8 * x - 16 * x**2
l11x = 8 - 32 * x
l12 = 8 * x**2 - 2 * x
l12x = 16 * x - 2
l20 = 8 * x**2 - 14 * x + 6
l20x = 16 * x - 14
l21 = -16 * x**2 + 24 * x - 8
l21x = -32 * x + 24
l22 = 8 * x**2 - 10 * x + 3
l22x = 16 * x - 10
peTsensBoundaryMat = np.zeros(
    (
        2 * ((model.nCollocation + 1) * model.nElements + 1),
        2 * (model.nCollocation + 1) * model.nElements + 2,
    )
)
peTsensBoundaryMat[0:5, 0:5] = np.array(
    [
        [l10x[0] - params["PeT"], l11x[0], l12x[0], 0, params["PeT"] * params["f"]],
        [0, 1, 0, 0, 0],
        [l10x[2], l11x[2], l12x[2] - l20x[2], -l21x[2], -l22x[2]],
        [0, 0, 0, 1, 0],
        [0, 0, l20x[4], l21x[4], l22x[4]],
    ]
)
peTsensBoundaryMat[5:, 5:] = peTsensBoundaryMat[0:5, 0:5]
peTsensBoundaryMat[5, 0] = -1
peTsensBoundaryMat[5, 4] = params["f"]
# Check Boundary matrices
# print(model.dvdPeTboundaryMat)
# print(peTsensBoundaryMat)
assert np.isclose(model.dvdPeTboundaryMat, peTsensBoundaryMat).all()
print("         PeT Sensitivity Boundary Matrix Passing")

print("     Testing dydt Computation")
# Check dydt computed values
params = {"PeM": 1, "PeT": 1, "f": 0, "Le": 1, "Da": 0, "beta": 0, "gamma": 0, "delta": 0, "vH": 0}
model = TankModel(nCollocation=3, nElements=2, spacing="legendre", bounds=[0, 1], params=params)
x = model.collocationPoints
nPoints = model.nCollocation * model.nElements
# Case 1a: u=x^2, uvH=x^2, v=x^2, vvH=x^2
order = 3
if order == 2:
    u = -(x**2) + 2 * x + 2 / params["PeM"]
    v = -(x**2) + 2 * x + (2 / params["PeT"] + params["f"] * (2 - 1)) / (1 - params["f"])
    dudx = -2 * x + 2
    dvdx = -2 * x + 2
    d2udx2 = -2
    d2vdx2 = -2
elif order == 3:
    u = -(x**3) - x**2 + 5 * x + 5 / params["PeT"]
    v = -(x**3) - x**2 + 5 * x + (5 / params["PeT"] + params["f"] * (5 - 2)) / (1 - params["f"])
    dudx = -3 * x**2 - 2 * x + 5
    dvdx = -3 * x**2 - 2 * x + 5
    d2udx2 = -6 * x - 2
    d2vdx2 = -6 * x - 2

if params["f"] == 1:
    v = -(x**2) + 2 * x + 1
else:
    v = -(x**3) - x**2 + 5 * x + (5 / params["PeT"] + params["f"] * (5 - 2)) / (1 - params["f"])
y = np.concatenate((u, v), axis=0)
dydt = model.dydt(y, 0)
assert np.isclose(
    dydt[:nPoints],
    -dudx
    + d2udx2 / params["PeM"]
    + params["Da"] * (1 - u) * np.exp(params["gamma"] * params["beta"] / (1 + params["beta"] * v)),
).all()
assert np.isclose(
    dydt[nPoints:],
    (
        -dvdx
        + d2vdx2 / params["PeT"]
        + params["Da"] * (1 - u) * np.exp(params["gamma"] * params["beta"] / (1 + params["beta"] * v))
        + params["delta"] * (params["vH"] - v)
    )
    / params["Le"],
).all()
print("         Advec-Diffusion Case Passing")


params = {
    "PeM": 1,
    "PeT": 1,
    "f": 0,
    "Le": 1,
    "Da": 0.15,
    "beta": 1.3,
    "gamma": 10,
    "delta": 2,
    "vH": -0.2,
}
model = TankModel(nCollocation=2, nElements=4, spacing="legendre", bounds=[0, 1], params=params)
x = model.collocationPoints
nPoints = model.nCollocation * model.nElements
# Case 1a: u=x^2, uvH=x^2, v=x^2, vvH=x^2
order = 3
if order == 2:
    u = -(x**2) + 2 * x + 2 / params["PeM"]
    v = -(x**2) + 2 * x + (2 / params["PeT"] + params["f"] * (2 - 1)) / (1 - params["f"])
    dudx = -2 * x + 2
    dvdx = -2 * x + 2
    d2udx2 = -2
    d2vdx2 = -2
elif order == 3:
    u = -(x**3) - x**2 + 5 * x + 5 / params["PeT"]
    v = -(x**3) - x**2 + 5 * x + (5 / params["PeT"] + params["f"] * (5 - 2)) / (1 - params["f"])
    dudx = -3 * x**2 - 2 * x + 5
    dvdx = -3 * x**2 - 2 * x + 5
    d2udx2 = -6 * x - 2
    d2vdx2 = -6 * x - 2

if params["f"] == 1:
    v = -(x**2) + 2 * x + 1
else:
    v = -(x**3) - x**2 + 5 * x + (5 / params["PeT"] + params["f"] * (5 - 2)) / (1 - params["f"])
    v = -(x**3) - x**2 + 5 * x + (5 / params["PeT"] + params["f"] * (5 - 2)) / (1 - params["f"])
y = np.concatenate((u, v), axis=0)
dydt = model.dydt(y, 0)
assert np.isclose(
    dydt[:nPoints],
    -dudx
    + d2udx2 / params["PeM"]
    + params["Da"] * (1 - u) * np.exp(params["gamma"] * params["beta"] * v / (1 + params["beta"] * v)),
).all()
assert np.isclose(
    dydt[nPoints:],
    (
        -dvdx
        + d2vdx2 / params["PeT"]
        + params["Da"] * (1 - u) * np.exp(params["gamma"] * params["beta"] * v / (1 + params["beta"] * v))
        + params["delta"] * (params["vH"] - v)
    )
    / params["Le"],
).all()
print("         Nonlinear Terms Passing")


params = {
    "PeM": 1000,
    "PeT": 100,
    "f": 0.3,
    "Le": 1,
    "Da": 0.15,
    "beta": 1.3,
    "gamma": 10,
    "delta": 2,
    "vH": -0.2,
}
model = TankModel(nCollocation=3, nElements=5, spacing="legendre", bounds=[0, 1], params=params)
x = model.collocationPoints
nPoints = model.nCollocation * model.nElements
# Case 1a: u=x^2, uvH=x^2, v=x^2, vvH=x^2
order = 3
if order == 2:
    u = -(x**2) + 2 * x + 2 / params["PeM"]
    v = -(x**2) + 2 * x + (2 / params["PeT"] + params["f"] * (2 - 1)) / (1 - params["f"])
    dudx = -2 * x + 2
    dvdx = -2 * x + 2
    d2udx2 = -2
    d2vdx2 = -2
elif order == 3:
    u = -(x**3) - x**2 + 5 * x + 5 / params["PeM"]
    v = -(x**3) - x**2 + 5 * x + (5 / params["PeT"] + params["f"] * (5 - 2)) / (1 - params["f"])
    dudx = -3 * x**2 - 2 * x + 5
    dvdx = -3 * x**2 - 2 * x + 5
    d2udx2 = -6 * x - 2
    d2vdx2 = -6 * x - 2

if params["f"] == 1:
    v = -(x**2) + 2 * x + 1
else:
    v = -(x**3) - x**2 + 5 * x + (5 / params["PeT"] + params["f"] * (5 - 2)) / (1 - params["f"])
y = np.concatenate((u, v), axis=0)
dydt = model.dydt(y, 0)


assert np.isclose(
    dydt[:nPoints],
    -dudx
    + d2udx2 / params["PeM"]
    + params["Da"] * (1 - u) * np.exp(params["gamma"] * params["beta"] * v / (1 + params["beta"] * v)),
).all()
assert np.isclose(
    dydt[nPoints:],
    (
        -dvdx
        + d2vdx2 / params["PeT"]
        + params["Da"] * (1 - u) * np.exp(params["gamma"] * params["beta"] * v / (1 + params["beta"] * v))
        + params["delta"] * (params["vH"] - v)
    )
    / params["Le"],
).all()
print("         Full Model Passing")


# Check Integration: Spatial and Temporal Integration
print("testTankModely.py passes")
