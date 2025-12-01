import os

import matplotlib.pyplot as plt
import numpy as np
import scipy

from tankModel.TankModel import TankModel


def bifurcationSearch(model, bifurcationParam, tEval, skipProp, qoi_type, y0):
    if qoi_type not in ("outflow_temp"):
        raise (Exception("Invalid qoi_type: ", qoi_type))
    # Parameters
    paramMin = bifurcationParam[1][0]
    paramMax = bifurcationParam[1][1]
    paramStep = bifurcationParam[1][2]
    max_iterations = tEval.size
    skip_iterations = max_iterations * skipProp
    max_counter = int((max_iterations - skip_iterations) * (((paramMax - paramMin) / paramStep) + 1))
    if y0.ndim == 1:
        y0 = np.array([y0])
    ny0 = y0.shape[0]
    # The x and r results will be stored in these two arrays
    result_qoi = np.zeros(max_counter * ny0)
    result_poi = np.zeros(max_counter * ny0)
    # Start the main loop
    i = 0
    for iy0 in range(y0.shape[0]):
        for r in np.arange(paramMin, paramMax, paramStep):
            modelCoeff = y0[iy0, :]
            # print("Starting parameter value: ", r)
            if bifurcationParam[0] == "vH":
                model.params["vH"] = r
            else:
                raise (Exception("Invalid bifurcation param: ", bifurcationParam[0]))
            for it in range(max_iterations - 1):

                def rhs(t, y):
                    return model.dydt(y, t)

                odeOut = scipy.integrate.solve_ivp(
                    rhs,
                    (tEval[it], tEval[it + 1]),
                    modelCoeff,
                    method="BDF",
                    atol=1e-13,
                    rtol=1e-13,
                )
                modelCoeff = odeOut.y[:, -1]
                if it > skip_iterations:
                    result_poi[i] = r
                    if qoi_type == "outflow_temp":
                        result_qoi[i] = model.eval(1, modelCoeff, output="v")
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


tEval = np.linspace(0, 200, 41)
bifurcationParam = ("vH", [-0.09, -0.02, 0.005])
skipProp = 0.5
qoi_type = "outflow_temp"


resultsFolder = "../../results/verification/"
saveLocation = resultsFolder + "/bifuractionResults_" + str(qoi_type) + "_" + str(bifurcationParam[0])
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
fig, ax = plt.subplots(figsize=(3.8, 3))
plt.plot(result_poi, result_qoi, ".", color="k")
plt.xlabel(r"$v_H$")
plt.ylabel(r"$v(t,1)$")
ax.set_xticks([-0.09, -0.07, -0.05, -0.03])
ax.set_yticks([-0.05, 0.05, 0.15, 0.25, 0.35, 0.45])
plt.tight_layout()
plt.savefig(saveLocation + "bifurcation.pdf", format="pdf")
plt.savefig(saveLocation + "bifurcation.png", format="png")
plt.show()
