#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   FEM/mms/tests/testTankMMS.py
@Time    :   2024/05/24 11:16:26
@Author  :   Harley Hanes
@Version :   1.0
@Contact :   hhanes@ncsu.edu
@License :   (C)Copyright 2024, Harley Hanes
@Desc    :   Unit tests for tankMMS.py
"""
import numpy as np

import mms.tankMMS as tankMMS
from tankModel.TankModel import TankModel

print("Running testTankMMS.py")
# Test 1: Convergence Rate testing
error = np.array([1, 1 / 2, (1 / 2) ** 3, (1 / 2) ** 6, (1 / 2) ** 10, (1 / 2) ** 15])
step_size = np.array([1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32])
convergenceRates = tankMMS.computeConvergenceRates(step_size, error)
assert np.all(np.isclose(convergenceRates, np.array([1.0, 2.0, 3.0, 4.0, 5.0])))
print("     Convergence Rate Test passes")

# Test 2: Check MMS functions are generated correclty
spatialOrder = 2
params = {"PeM": 1, "PeT": 1, "f": 5, "Le": 0, "Da": 0, "beta": 0, "gamma": 0, "delta": 0, "vH": 0}


def temporal(t):
    return 1 + 1 * t


def temporaldt(t):
    return 1 + 0 * t


u, dudt, dudx, dudx2, v, dvdt, dvdx, dvdx2 = tankMMS.constructPolynomialMMSsolutionFunction(
    spatialOrder, params, temporal, temporaldt
)


# Nominal Value tests
def uExpected(t, x):
    return temporal(t) * (-(x**2) + 2 * x + 2 / params["PeM"])


def vExpected(t, x):
    return temporal(t) * (-(x**2) + 2 * x + (2 / params["PeT"] + params["f"]) / (1 - params["f"]))


assert uExpected(0, 0) == u(0, 0)
assert vExpected(0, 0) == v(0, 0)
assert uExpected(0, 1) == u(0, 1)
assert vExpected(0, 1) == v(0, 1)
assert uExpected(0, 2) == u(0, 2)
assert vExpected(0, 2) == v(0, 2)
assert uExpected(1, 0) == u(1, 0)
assert vExpected(1, 0) == v(1, 0)


# x-derivative tests
def dudxExpected(t, x):
    return temporal(t) * (-2 * x + 2)


def dvdxExpected(t, x):
    return temporal(t) * (-2 * x + 2)


assert dudxExpected(0, 0) == dudx(0, 0)
assert dvdxExpected(0, 0) == dvdx(0, 0)
assert dudxExpected(0, 1) == dudx(0, 1)
assert dvdxExpected(0, 1) == dvdx(0, 1)
assert dudxExpected(1, 0) == dudx(1, 0)
assert dvdxExpected(1, 0) == dvdx(1, 0)


# dx2 tests
def dudx2Expected(t, x):
    return temporal(t) * (-2 + 0 * x)


def dvdx2Expected(t, x):
    return temporal(t) * (-2 + 0 * x)


assert dudx2Expected(0, 0) == dudx2(0, 0)
assert dvdx2Expected(0, 0) == dvdx2(0, 0)
assert dudx2Expected(0, 1) == dudx2(0, 1)
assert dvdx2Expected(0, 1) == dvdx2(0, 1)
assert dudx2Expected(1, 0) == dudx2(1, 0)
assert dvdx2Expected(1, 0) == dvdx2(1, 0)


# dt tests
def dudtExpected(t, x):
    return temporaldt(t) * (-(x**2) + 2 * x + 2 / params["PeM"])


def dvdtExpected(t, x):
    return temporaldt(t) * (-(x**2) + 2 * x + (2 / params["PeT"] + params["f"]) / (1 - params["f"]))


assert dudtExpected(0, 0) == dudt(0, 0)
assert dvdtExpected(0, 0) == dvdt(0, 0)
assert dudtExpected(0, 1) == dudt(0, 1)
assert dvdtExpected(0, 1) == dvdt(0, 1)
assert dudtExpected(0, 2) == dudt(0, 2)
assert dvdtExpected(0, 2) == dvdt(0, 2)
assert dudtExpected(1, 0) == dudt(1, 0)
assert dvdtExpected(1, 0) == dvdt(1, 0)
print("     MMS Solution Test passes")

# Test 3: Check errors are computed correctly
model = TankModel(nCollocation=3, nElements=2, spacing="legendre", bounds=[0, 1])


def f1(x):
    return np.outer(np.array([1, 2, 3]), x**2).squeeze()


def f2(x):
    return np.outer(np.array([1, 2, 3]), x).squeeze()


def errorFunction(x):
    return f1(x) - f2(x)


def squaredErrorFunction(x):
    return (f1(x) - f2(x)) ** 2


def squaredReferenceFunction(x):
    return f2(x) ** 2


errorL2, errorL2space = tankMMS.computeL2error(
    model, squaredErrorFunction, squaredReferenceFunction, np.array([1, 2, 3])
)

expectedErrorL2space = np.array([1 / np.sqrt(10), 1 / np.sqrt(10), 1 / np.sqrt(10)])
expectedErrorL2 = 1 / np.sqrt(10)

assert np.isclose(expectedErrorL2space, errorL2space).all()
assert np.isclose(expectedErrorL2, errorL2)

errorLinf, errorLinfSpace = tankMMS.computeLinfError(errorFunction(np.linspace(0, 1, 101)), f2(np.linspace(0, 1, 101)))
print(errorLinfSpace)
assert np.isclose(errorLinfSpace, np.array([1 / 4, 1 / 4, 1 / 4])).all()
assert np.isclose(errorLinf, 1 / 4)


print("testTankMMS.py passes")
