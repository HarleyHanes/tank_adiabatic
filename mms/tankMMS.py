import numpy as np
import scipy

from tankModel.TankModel import TankModel


def runMMStest(higherOrders, nCollocations, nElems, xEval, tEval, params, verbosity=0):
    # Parse inputs
    def temporal0(t):
        return 1 + 0 * t

    def temporal1(t):
        return 1 + 1 * t

    def temporal_exp(t):
        return 1 + t * np.exp(2 * t)

    temporals = [temporal0, temporal1, temporal_exp]

    def temporaldt0(t):
        return 0 * t

    def temporaldt1(t):
        return 1 + 0 * t

    def temporaldt_exp(t):
        return np.exp(2 * t) + 2 * t * np.exp(2 * t)

    temporalsdt = [temporaldt0, temporaldt1, temporaldt_exp]
    # temporals = [lambda t: 1+0*t, lambda t: t, lambda t: t**2, lambda t: np.sin(t)]
    # temporalsdt = [lambda t: 0*t, lambda t: 1+0*t, lambda t: 2*t, lambda t: np.cos(t)]
    # Pre-allocate solution and error arrays.
    # Indices (2,2) are for (u,v) and then (MMS, Model)
    error = np.empty((len(nCollocations), len(nElems), len(temporals), len(higherOrders), 2, 2))
    errorSpace = np.empty((len(nCollocations), len(nElems), len(temporals), len(higherOrders), 2, 2, tEval.size))
    solutions = np.empty(
        (
            len(nCollocations),
            len(nElems),
            len(temporals),
            len(higherOrders),
            2,
            2,
            tEval.size,
            xEval.size,
        )
    )
    jointConvergenceRates = np.empty((len(nCollocations), len(nElems) - 1, len(temporals), len(higherOrders), 2, 2))
    spatialConvergenceRates = np.empty(
        (len(nCollocations), len(nElems) - 1, len(temporals), len(higherOrders), 2, 2, tEval.size)
    )
    for iColl in range(len(nCollocations)):
        for iElem in range(len(nElems)):
            if verbosity > 0:
                print(
                    "Testing for (collocation points, elements): ("
                    + str(nCollocations[iColl])
                    + ", "
                    + str(nElems[iElem])
                    + ")"
                )
            # Generate tankModel
            model = TankModel(
                nCollocation=nCollocations[iColl],
                nElements=nElems[iElem],
                spacing="legendre",
                bounds=[0, 1],
                params=params,
                verbosity=verbosity,
            )
            # Read back params so that boundary Peclets are saved if entered params vector did not include them
            params = model.params
            # Set discretization
            # Loop through exact monomial cases
            # Loop through temporal functions
            for itemporal in range(len(temporals)):
                if verbosity > 0:
                    print("iTemporal: ", itemporal)
                for iorder in range(len(higherOrders)):
                    if isinstance(higherOrders[iorder], int):
                        spatialOrder = higherOrders[iorder]
                        if verbosity > 0:
                            print("Order: ", spatialOrder)
                        # Construct solutions that are sum of monomials up to spatialOrder
                        u, dudt, dudx, dudx2, v, dvdt, dvdx, dvdx2 = constructPolynomialMMSsolutionFunction(
                            spatialOrder, params, temporals[itemporal], temporalsdt[itemporal]
                        )
                    if isinstance(higherOrders[iorder], str):
                        if higherOrders[iorder] == "sin":
                            u, dudt, dudx, dudx2, v, dvdt, dvdx, dvdx2 = constructSinMMssolutionFunction(
                                params, temporals[itemporal], temporalsdt[itemporal]
                            )

                    def u_cp(t):
                        return u(t, model.collocationPoints)

                    def dudt_cp(t):
                        return dudt(t, model.collocationPoints)

                    def dudx_cp(t):
                        return dudx(t, model.collocationPoints)

                    def dudx2_cp(t):
                        return dudx2(t, model.collocationPoints)

                    def v_cp(t):
                        return v(t, model.collocationPoints)

                    def dvdt_cp(t):
                        return dvdt(t, model.collocationPoints)

                    def dvdx_cp(t):
                        return dvdx(t, model.collocationPoints)

                    def dvdx2_cp(t):
                        return dvdx2(t, model.collocationPoints)

                    sourceFunction = constructSourceTermFunction(
                        u_cp,
                        dudt_cp,
                        dudx_cp,
                        dudx2_cp,
                        v_cp,
                        dvdt_cp,
                        dvdx_cp,
                        dvdx2_cp,
                        params,
                    )

                    y0 = np.append(u(tEval[0], model.collocationPoints), v(tEval[0], model.collocationPoints))

                    # Optional: quick sanity plot (disabled)
                    # if itemporal == 0:
                    #     uErr = model.eval(xEval, y0, output="u") - u(0, xEval)
                    #     uRef = u(0, xEval)
                    #     plt.semilogy(xEval, np.sqrt(uErr**2 / (uRef**2)))
                    #     plt.legend()
                    # Compute Model Coeff
                    modelCoeff = np.empty((tEval.size, y0.size))
                    modelCoeff[0] = y0
                    for i in range(modelCoeff.shape[0] - 1):

                        def rhs(t, y):
                            return model.dydtSource(y, t, sourceFunction)

                        odeOut = scipy.integrate.solve_ivp(
                            rhs,
                            (tEval[i], tEval[i + 1]),
                            modelCoeff[i],
                            method="BDF",
                            atol=1e-13,
                            rtol=1e-13,
                        )
                        modelCoeff[i + 1] = odeOut.y[:, -1]
                        if odeOut.status != 0:
                            print("Warning: ode solver terminated prematurely")
                    # Check for error between model coeffecients at t=0 as outputed by ODE function and the true
                    y0error = np.max(modelCoeff[0, :] - y0)
                    if y0error > 10 ** (-14):
                        print("Warning: odeint has non-zero error in y0")
                        print("y0 error: %03e" % (y0error,))

                    # Evalute MMS and model at plot points
                    solutions[iColl, iElem, itemporal, iorder, 0, 0] = u(tEval, xEval)
                    solutions[iColl, iElem, itemporal, iorder, 0, 1] = model.eval(xEval, modelCoeff, output="u")
                    solutions[iColl, iElem, itemporal, iorder, 1, 0] = v(tEval, xEval)
                    solutions[iColl, iElem, itemporal, iorder, 1, 1] = model.eval(xEval, modelCoeff, output="v")

                    # Compute Error
                    def uErrorFunction(x):
                        return model.eval(x, modelCoeff, output="u") - u(tEval, x)

                    def vErrorFunction(x):
                        return model.eval(x, modelCoeff, output="v") - v(tEval, x)

                    def uSquaredErrorFunction(x):
                        return uErrorFunction(x) ** 2

                    def vSquaredErrorFunction(x):
                        return vErrorFunction(x) ** 2

                    def uSquaredReferenceFunction(x):
                        return u(tEval, x) ** 2

                    def vSquaredReferenceFunction(x):
                        return v(tEval, x) ** 2

                    # print(uSquaredErrorFunction(xEval).shape)
                    # print(uErrorFunction(xEval)[0,:])
                    # print(uSquaredReferenceFunction(xEval)[0,:])
                    quadOrder = spatialOrder**2
                    # quadOrder="auto"
                    # print("quadOrder: ", quadOrder)
                    uErrorL2, uErrorL2space = computeL2error(
                        model,
                        uSquaredErrorFunction,
                        uSquaredReferenceFunction,
                        tEval,
                        order=quadOrder,
                    )
                    vErrorL2, vErrorL2space = computeL2error(
                        model,
                        vSquaredErrorFunction,
                        vSquaredReferenceFunction,
                        tEval,
                        order=quadOrder,
                    )
                    uErrorLinf, uErrorLinfSpace = computeLinfError(uErrorFunction(xEval), u(tEval, xEval))
                    vErrorLinf, vErrorLinfSpace = computeLinfError(vErrorFunction(xEval), u(tEval, xEval))
                    # print(np.mean(np.sqrt(uSquaredErrorFunction(xEval)[0,:]/uSquaredReferenceFunction(xEval)[0,:])))
                    # print(uErrorL2space[0])
                    error[iColl, iElem, itemporal, iorder, 0, 0] = uErrorL2
                    error[iColl, iElem, itemporal, iorder, 0, 1] = uErrorLinf
                    error[iColl, iElem, itemporal, iorder, 1, 0] = vErrorL2
                    error[iColl, iElem, itemporal, iorder, 1, 1] = vErrorLinf
                    errorSpace[iColl, iElem, itemporal, iorder, 0, 0, :] = uErrorL2space
                    errorSpace[iColl, iElem, itemporal, iorder, 0, 1, :] = uErrorLinfSpace
                    errorSpace[iColl, iElem, itemporal, iorder, 1, 0, :] = vErrorL2space
                    errorSpace[iColl, iElem, itemporal, iorder, 1, 1, :] = vErrorLinfSpace

        spatialConvergenceRates[iColl] = computeConvergenceRates(1 / np.array(nElems), errorSpace[iColl])
        jointConvergenceRates[iColl] = computeConvergenceRates(1 / np.array(nElems), error[iColl])

    return error, solutions, jointConvergenceRates, errorSpace, spatialConvergenceRates


def computeConvergenceRates(discretizations, errors):
    # Check discretizations and errors have the same dimensions or that discretizations is 1D
    if discretizations.ndim == 1:
        assert discretizations.shape[0] == errors.shape[0]
    else:
        assert discretizations.shape == errors.shape
    # Define convergence rates as having the same shape as discretizations but with 1 less in the first dimension
    convergenceRates = np.log(errors[0:-1].T / errors[1:].T) / np.log(discretizations[0:-1] / discretizations[1:])
    return convergenceRates.T


def constructPolynomialMMSsolutionFunction(spatialOrder, params, temporal, temporaldt):
    uSpatialCoeff = -np.ones((spatialOrder + 1,))
    vSpatialCoeff = -np.ones((spatialOrder + 1,))

    uSpatialCoeff[1] = np.dot(np.arange(2, spatialOrder + 1), np.ones((spatialOrder - 1)))
    vSpatialCoeff[1] = np.dot(np.arange(2, spatialOrder + 1), np.ones((spatialOrder - 1)))
    uSpatialCoeff[0] = uSpatialCoeff[1] / params["PeM-boundary"]
    vSpatialCoeff[0] = (
        vSpatialCoeff[1] / params["PeT-boundary"] + params["f"] * (vSpatialCoeff[1] - (spatialOrder - 1))
    ) / (1 - params["f"])

    dudxSpatialCoeff = uSpatialCoeff[1:] * np.arange(1, spatialOrder + 1)
    dvdxSpatialCoeff = vSpatialCoeff[1:] * np.arange(1, spatialOrder + 1)

    dudx2SpatialCoeff = dudxSpatialCoeff[1:] * np.arange(1, spatialOrder)
    dvdx2SpatialCoeff = dvdxSpatialCoeff[1:] * np.arange(1, spatialOrder)

    # print("uSpatialCoeff: ", uSpatialCoeff)
    # print("vSpatialCoeff: ", vSpatialCoeff)
    # print("dudxSpatialCoeff: ", dudxSpatialCoeff)
    # print("dudxSpatialCoeff: ", dvdxSpatialCoeff)
    # print("dudx2SpatialCoeff: ", dudx2SpatialCoeff)
    # print("dudx2SpatialCoeff: ", dvdx2SpatialCoeff)

    def u(t, x):
        return np.outer(
            temporal(t),
            np.sum(np.power.outer(x, np.arange(0, spatialOrder + 1)) * uSpatialCoeff, axis=-1),
        ).squeeze()

    def v(t, x):
        return np.outer(
            temporal(t),
            np.sum(np.power.outer(x, np.arange(0, spatialOrder + 1)) * vSpatialCoeff, axis=-1),
        ).squeeze()

    def dudx(t, x):
        return np.outer(
            temporal(t),
            np.sum(np.power.outer(x, np.arange(0, spatialOrder)) * dudxSpatialCoeff, axis=-1),
        ).squeeze()

    def dvdx(t, x):
        return np.outer(
            temporal(t),
            np.sum(np.power.outer(x, np.arange(0, spatialOrder)) * dvdxSpatialCoeff, axis=-1),
        ).squeeze()

    def dudx2(t, x):
        return np.outer(
            temporal(t),
            np.sum(np.power.outer(x, np.arange(0, spatialOrder - 1)) * dudx2SpatialCoeff, axis=-1),
        ).squeeze()

    def dvdx2(t, x):
        return np.outer(
            temporal(t),
            np.sum(np.power.outer(x, np.arange(0, spatialOrder - 1)) * dvdx2SpatialCoeff, axis=-1),
        ).squeeze()

    def dudt(t, x):
        return np.outer(
            temporaldt(t),
            np.sum(np.power.outer(x, np.arange(0, spatialOrder + 1)) * uSpatialCoeff, axis=-1),
        ).squeeze()

    def dvdt(t, x):
        return np.outer(
            temporaldt(t),
            np.sum(np.power.outer(x, np.arange(0, spatialOrder + 1)) * vSpatialCoeff, axis=-1),
        ).squeeze()

    return u, dudt, dudx, dudx2, v, dvdt, dvdx, dvdx2


def constructSinMMssolutionFunction(params, temporal, temporaldt):
    freq = np.pi
    weight = 1

    linearCoeff = -freq * weight * np.cos(freq)
    uConstCoeff = 1 / params["PeM-boundary"] * (freq * weight + linearCoeff)
    vConstCoeff = (
        (freq * weight + linearCoeff) / params["PeT-boundary"] + params["f"] * (weight * np.sin(freq) + linearCoeff)
    ) / (1 - params["f"])

    def u(t, x):
        return np.outer(temporal(t), weight * np.sin(freq * x) + linearCoeff * x + uConstCoeff).squeeze()

    def v(t, x):
        return np.outer(temporal(t), weight * np.sin(freq * x) + linearCoeff * x + vConstCoeff).squeeze()

    def dudt(t, x):
        return np.outer(temporaldt(t), weight * np.sin(freq * x) + linearCoeff * x + uConstCoeff).squeeze()

    def dvdt(t, x):
        return np.outer(temporaldt(t), weight * np.sin(freq * x) + linearCoeff * x + vConstCoeff).squeeze()

    def dudx(t, x):
        return np.outer(temporal(t), freq * weight * np.cos(freq * x) + linearCoeff).squeeze()

    def dvdx(t, x):
        return np.outer(temporal(t), freq * weight * np.cos(freq * x) + linearCoeff).squeeze()

    def dudx2(t, x):
        return np.outer(temporal(t), -(freq**2) * weight * np.sin(freq * x)).squeeze()

    def dvdx2(t, x):
        return np.outer(temporal(t), -(freq**2) * weight * np.sin(freq * x)).squeeze()

    return u, dudt, dudx, dudx2, v, dvdt, dvdx, dvdx2


def constructSourceTermFunction(u, dudt, dudx, dudx2, v, dvdt, dvdx, dvdx2, params):
    # Construct source term as functions of time t
    def sourceU(t):
        return (
            dudt(t)
            + dudx(t)
            - (
                dudx2(t) / params["PeM"]
                + params["Da"]
                * (1 - u(t))
                * np.exp(params["gamma"] * params["beta"] * v(t) / (1 + params["beta"] * v(t)))
            )
        )

    def sourceV(t):
        return (
            params["Le"] * dvdt(t)
            + dvdx(t)
            - (
                dvdx2(t) / params["PeT"]
                + params["Da"]
                * (1 - u(t))
                * np.exp(params["gamma"] * params["beta"] * v(t) / (1 + params["beta"] * v(t)))
                + params["delta"] * (params["vH"] - v(t))
            )
        )

    def source(t):
        return np.concatenate((sourceU(t), sourceV(t)), axis=-1)

    return source


#!!!!!!!!!Change all of these to be (time,space) for consistency elsewhere in code
def computeL2error(model, squaredErrorFunction, squaredReferenceFunction, tPoints, order="auto"):
    errorL2Squared, errorL2SpaceSquared = model.integrate(
        squaredErrorFunction, tPoints, integrateTime=True, order=order
    )
    # print("Pre-relatives L2 error: ", errorL2SpaceSquared)
    referenceL2Squared, referenceL2SpaceSquared = model.integrate(
        squaredReferenceFunction, tPoints, integrateTime=True, order=order
    )
    errorL2 = np.sqrt(errorL2Squared / referenceL2Squared)
    errorL2Space = np.sqrt(errorL2SpaceSquared / referenceL2SpaceSquared)
    return errorL2, errorL2Space


def computeLinfError(error, reference):
    errorLinfSpace = np.max(np.abs(error), axis=1) / np.max(np.abs(reference), axis=1)
    errorLinf = np.max(np.abs(error)) / np.max(np.abs(reference))
    return errorLinf, errorLinfSpace
