import os

import matplotlib.pyplot as plt
import numpy as np
import scipy

from tankModel.TankModel import TankModel


def bifurcationSearch(model, bifurcationParam, tEval, skipProp, qoi_type, y0):
    if qoi_type not in ("outflow_temp", "min/max temp"):
        raise (Exception("Invalid qoi_type: ", qoi_type))
    # Parameters
    paramMin = bifurcationParam[1][0]
    paramMax = bifurcationParam[1][1]
    paramStep = bifurcationParam[1][2]
    max_iterations = tEval.size
    skip_iterations = int(max_iterations * skipProp)
    max_counter = int((max_iterations - skip_iterations) * (((paramMax - paramMin) / paramStep) + 1))
    paramSet = np.arange(paramMin, paramMax, paramStep)
    if y0.ndim == 1:
        y0 = np.array([y0])
    ny0 = y0.shape[0]
    # The x and r results will be stored in these two arrays
    result_qoi = np.zeros(max_counter * ny0)
    result_poi = np.zeros(max_counter * ny0)
    # Start the main loop
    i = 0
    for iy0 in range(y0.shape[0]):
        modelCoeff = y0[iy0, :]
        for r in paramSet:
            if modelCoeff.ndim != 1:
                modelCoeff = modelCoeff[skip_iterations, :]
            # print("Starting parameter value: ", r)
            if bifurcationParam[0] == "vH":
                model.params["vH"] = r
            else:
                raise (Exception("Invalid bifurcation param: ", bifurcationParam[0]))

            def rhs(t, y):
                return model.dydt(y, t)

            odeOut = scipy.integrate.solve_ivp(
                rhs,
                (tEval[0], tEval[-1]),
                # modelCoeff,
                y0[iy0, :],
                t_eval=tEval,
                method="BDF",
                atol=1e-13,
                rtol=1e-13,
            )
            modelCoeff = odeOut.y.transpose()
            if qoi_type == "outflow_temp":
                raise ValueError("outflow_temp deprecated")
            elif qoi_type == "min/max temp":
                outletTemp = model.eval(1, modelCoeff[skip_iterations:-1, :], output="v")
                result_poi[i] = r
                result_qoi[i] = np.min(outletTemp)
                i += 1
                result_poi[i] = r
                result_qoi[i] = np.max(outletTemp)
                i += 1

    result_qoi = result_qoi[result_poi != 0].copy()
    result_poi = result_poi[result_poi != 0].copy()
    return result_qoi, result_poi


plotFromData = True
if not plotFromData:
    baseParams = {
        "PeM": 300,
        "PeT": 300,
        "f": 0.3,
        "Le": 1,
        "Da": 0.15,
        "beta": 1.4,
        "gamma": 10,
        "delta": 2,
        "vH": -0.03,
    }
    model = TankModel(nCollocation=2, nElements=64, spacing="legendre", bounds=[0, 1], params=baseParams)
    modelCoeff = 2 * np.ones((2, model.nCollocation * model.nElements * 2))
    tEval = np.linspace(0, 200, 26)
    for it in range(tEval.size - 1):

        def rhs(t, y):
            return model.dydt(y, t)

        odeOut = scipy.integrate.solve_ivp(
            rhs,
            (tEval[it], tEval[it + 1]),
            modelCoeff[0],
            method="BDF",
            atol=1e-13,
            rtol=1e-13,
        )
        modelCoeff[0] = odeOut.y[:, -1]

    baseParams = {
        "PeM": 300,
        "PeT": 300,
        "f": 0.3,
        "Le": 1,
        "Da": 0.15,
        "beta": 1.4,
        "gamma": 10,
        "delta": 2,
        "vH": -0.09,
    }
    model = TankModel(nCollocation=2, nElements=64, spacing="legendre", bounds=[0, 1], params=baseParams)
    tEval = np.linspace(0, 200, 26)
    for it in range(tEval.size - 1):

        def rhs(t, y):
            return model.dydt(y, t)

        odeOut = scipy.integrate.solve_ivp(
            rhs,
            (tEval[it], tEval[it + 1]),
            modelCoeff[1],
            method="BDF",
            atol=1e-13,
            rtol=1e-13,
        )
        modelCoeff[1] = odeOut.y[:, -1]
    # plot initial condition
    xPlot = np.linspace(0, 1, 200)
    tempValue = model.eval(xPlot, modelCoeff, output="v")
    plt.plot(xPlot, tempValue.transpose())


tEval = np.linspace(0, 250, 1001)
bifurcationParam = ("vH", [-0.09, -0.019999, 0.001])
skipProp = 0.75
qoi_type = "min/max temp"


resultsFolder = "../../results/verification/"
saveLocation = resultsFolder + "bifurcationResults_" + str(qoi_type) + "_" + str(bifurcationParam[0])
if plotFromData:
    bifurcationData = np.load(saveLocation + "bifurcationData.npz")
    result_qoi = bifurcationData["result_qoi"]
    result_poi = bifurcationData["result_poi"]
else:
    print("Starting Bifurcation Analysis")
    result_qoi, result_poi = bifurcationSearch(model, bifurcationParam, tEval, skipProp, qoi_type, modelCoeff)
    if not os.path.exists(saveLocation):
        os.makedirs(saveLocation)
    np.savez(saveLocation + "bifurcationData.npz", result_qoi=result_qoi, result_poi=result_poi)
# Plot
fig, ax = plt.subplots(figsize=(3.4 * 1.12, 3 * 1.14))
plt.plot(result_poi, result_qoi, ".", color="k")
plt.xlabel(r"$v_H$")
plt.ylabel(r"$v_{ex}(1)$", labelpad=-5)
ax.set_xticks([-0.09, -0.07, -0.05, -0.03])
ax.set_yticks([-0.05, 0.05, 0.15, 0.25, 0.35, 0.45])
ax.set_ylim([-0.05, 0.45])
ax.set_xlim([-0.09, -0.02])
# Add dashed grid lines at each major x/y tick
ax.grid(True, which="major", axis="both", linestyle="--")
plt.tight_layout()
plt.savefig(saveLocation + "bifurcation.pdf", format="pdf")
plt.savefig(saveLocation + "bifurcation.png", format="png")
plt.show()
