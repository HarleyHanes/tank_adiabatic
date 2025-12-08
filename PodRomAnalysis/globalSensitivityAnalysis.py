import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from PodRomAnalysis.podRomAnalysis import (
    computeInitialCondition,
    computeSensitivity,
    constructGlobalParameterSamples,
    getParameterOptions,
    getSensitivityOptions,
    mapROMdataToFOMspace,
)
from postProcessing.plot import plotErrorConvergence, plotRomMatrices, subplot
from tankModel.TankModel import TankModel


def main():
    # Define simulation details
    verbosity = 1
    showPlots = True
    # Run Types
    plotConvergence = False

    plotSensitivity = True
    plotRomInterpolation = False

    plotTimeSeries = False
    plotModes = False
    plotError = False
    plotRomCoeff = False
    plotSingularValues = False
    plotFullSpectra = False

    # FOM parameters
    paramSet = "BizonChaotic"  # BizonPeriodic, BizonLinear, BizonChaotic, BizonAdvecDiffusion
    nCollocation = 2
    nElements = 64
    odeMethod = "BDF"  # LSODA, BDF, Note: Need a stiff solver, LSODA fastest but BDF needed to support complex step
    nPoints = 599
    nT = 600

    # Parameter Sampling

    gsaMethod = "DGSM"
    if gsaMethod == "DGSM":
        equationSet = "nonBoundaryParams-noDa"
        nRomSamples = 20
        nFomSamples = 6
        paramBounding = 0.2  # Percentage around base value
    else:
        equationSet = "tankOnly"

    # ROM parameters
    usePodRom = True
    useEnergyThreshold = True
    nDeimPoints = "max"  # Base value for DEIM, max or integer
    nonLinReduction = 4.0  # Base value for nonLinReduction, 1 means no reduction
    penaltyStrength = 0
    sensInit = ["zero"]
    quadRule = ["gauss-legendre"]  # simpson, gauss-legendre, uniform, monte carlo
    mean_reduction = ["mean"]
    error_norm = [r"$L_2$ Error", r"$L_\infty$ Error"]
    qois = [
        "Max Reactivity",
        "Average Reactivity",
        "Max Outlet Temperature",
        "Average Outlet Temperature",
    ]
    romSensitivityApproach = [
        "sensEq",
        "complex",
    ]  # none, finite, sensEq, complex, only used if equationSet!=tankOnly
    finiteDelta = 1e-6  # Only used if equationSet!=tankOnly and romSensitivityApproach=="finite"
    complexDelta = 1e-9  # Only used if equationSet!=tankOnly and romSensitivityApproach=="complex"

    # Set simulation parameters
    # Set POD Retention
    if useEnergyThreshold:
        modeRetention = 0.995
    else:
        if plotConvergence:
            if paramSet == "BizonChaotic":
                modeRetention = list(range(6, 53))  # 1e-1 to 1e-5
                # modeRetention = list(range(6,39)) #1e-1 to 1e-4
                # modeRetention = list(range(6,89))
            elif paramSet == "BizonPeriodic":
                modeRetention = list(range(5, 32))  # 1e-1 to 1e-5
                # modeRetention = list(range(6,29))
            elif paramSet == "BizonLinear":
                # modeRetention = list(range(1,15))
                # modeRetention = list(range(2,15)) #1e-1 to 1e-5
                modeRetention = list(range(2, 20))  # 1e-1 to 1e-5
            elif paramSet == "BizonNonLinear":
                modeRetention = list(range(7, 59))
        else:
            modeRetention = [27]

    # Change all POD-ROM parameters to lists if not already
    if isinstance(modeRetention, (float, int)):
        modeRetention = [modeRetention]
    if isinstance(mean_reduction, str):
        mean_reduction = [mean_reduction]
    if equationSet != "tankOnly":
        if isinstance(romSensitivityApproach, str):
            romSensitivityApproach = [romSensitivityApproach]
        if isinstance(sensInit, str):
            sensInit = [sensInit]
    else:
        romSensitivityApproach = ["none"]
        sensInit = ["none"]
    if isinstance(quadRule, str):
        quadRule = [quadRule]

    # Get Parameter Settings
    baseParams, stabalized = getParameterOptions(paramSet)

    fomSaveFolder = "../../results/sensitivityAnalysis/" + paramSet

    fomSaveFolder += (
        "_nCol"
        + str(nCollocation)
        + "_nElem"
        + str(nElements)
        + "_nT"
        + str(nT)
        + "_nX"
        + str(nPoints)
        + "_"
        + odeMethod
        + "/"
        + equationSet
    )
    if not os.path.exists(fomSaveFolder):
        os.makedirs(fomSaveFolder)
    # Simulation Settings
    bounds = [0, 1]
    if stabalized:
        stabalizationTime = 150
        tmax = 4.1
    else:
        tmax = 1.5
    tEval = np.linspace(0, tmax, num=nT)
    # Determine parameters to get sensitivity of
    neq, paramSelect, uLabels, vLabels, combinedLabels = getSensitivityOptions(equationSet)
    # Construct parameter samples
    if paramSelect == []:
        fomParamSamples = [baseParams]
        romParamSamples = [baseParams]
    else:
        fomParamSamples, romParamSamples = constructGlobalParameterSamples(
            baseParams, paramSelect, paramBounding, nFomSamples, nRomSamples, "saltelli"
        )

    # Setup system
    if verbosity >= 1:
        print("Setting up system")
    model = TankModel(
        nCollocation=nCollocation,
        nElements=nElements,
        spacing="legendre",
        bounds=bounds,
        params=baseParams,
        tEval=tEval,
        odeMethod=odeMethod,
        verbosity=verbosity,
    )

    def dydtStabilization(t, y):
        return model.dydtSens(y, t, paramSelect=[])

    # ------------------------ Run Stabilization ------------------------
    if verbosity >= 1:
        print("Running Stabalization")
    if stabalized:
        # Run out till stabalizing in periodic domain
        initialCondition = np.zeros(model.nCollocation * model.nElements * 2 * neq)
        # Note: Sensitivities have 0 initial condition
        initialCondition[: model.nCollocation * model.nElements * 2] = model.solve_ivp(
            dydtStabilization,
            initialCondition[: model.nCollocation * model.nElements * 2],
            tEval=[0, stabalizationTime],
        )[-1, :]
    else:
        initialCondition = computeInitialCondition(model, neq)
    # ===== Get Simulation Data =====
    # Step 1: Get FOM Data that will be used to generate ROM
    if verbosity >= 1:
        print("Getting Simulation Data")
    dataModelCoeff = np.empty((len(fomParamSamples), nT, model.nCollocation * model.nElements * 2 * neq))
    for i in range(len(fomParamSamples)):
        perturbedModel = model.copy(params=fomParamSamples[i])

        def dydtSens(y, t):
            return perturbedModel.dydtSens(y, t, paramSelect=paramSelect)

        dataModelCoeff[i] = model.solve_ivp(lambda t, y: dydtSens(y, t), initialCondition)
        if verbosity >= 3:
            print("dataModelCoeff shape: ", dataModelCoeff.shape)

    # Step 2: Get FOM Data that will be used to compare to ROM
    refModelCoeff = np.empty((len(romParamSamples), nT, model.nCollocation * model.nElements * 2 * neq))
    for i in range(len(romParamSamples)):
        perturbedModel = model.copy(params=romParamSamples[i])

        def dydtSens(y, t):
            return perturbedModel.dydtSens(y, t, paramSelect=paramSelect)

        refModelCoeff[i] = model.solve_ivp(lambda t, y: dydtSens(y, t), initialCondition)
        if verbosity >= 3:
            print("refModelCoeff shape: ", refModelCoeff.shape)
    # ===== Run POD-ROM =====

    for iquad in range(len(quadRule)):
        for isens in range(len(romSensitivityApproach)):
            for iInit in range(len(sensInit)):
                if verbosity >= 1:
                    print(
                        f"Using {quadRule[iquad]} quadrature rule, "
                        f"{romSensitivityApproach[isens]} sensitivity approach, "
                        f"{sensInit[iInit]} sensitivity initialization"
                    )
                for imean in range(len(mean_reduction)):
                    truncationError = np.empty((len(modeRetention),))
                    error = []
                    qoiResults = []
                    x, W = model.getQuadWeights(nPoints, quadRule[iquad])
                    uFomData = np.empty((len(fomParamSamples), neq, nT, nPoints))
                    vFomData = np.empty((len(fomParamSamples), neq, nT, nPoints))
                    uRefData = np.empty((len(romParamSamples), neq, nT, nPoints))
                    vRefData = np.empty((len(romParamSamples), neq, nT, nPoints))
                    for i in range(neq):
                        fomStart = i * (2 * model.nCollocation * model.nElements)
                        for j in range(len(fomParamSamples)):
                            uFomData[j, i], vFomData[j, i] = model.eval(
                                x,
                                dataModelCoeff[
                                    j,
                                    :,
                                    fomStart : fomStart + 2 * model.nCollocation * model.nElements,
                                ],
                                output="seperated",
                            )
                        for j in range(len(romParamSamples)):
                            uRefData[j, i], vRefData[j, i] = model.eval(
                                x,
                                refModelCoeff[
                                    j,
                                    :,
                                    fomStart : fomStart + 2 * model.nCollocation * model.nElements,
                                ],
                                output="seperated",
                            )
                    for iret in range(len(modeRetention)):
                        # ============================= Construct POD
                        if usePodRom:
                            if equationSet != "tankOnly":
                                podSaveFolder = (
                                    fomSaveFolder + "_" + romSensitivityApproach[isens] + "_" + sensInit[iInit]
                                )
                                if romSensitivityApproach[isens] == "finite":
                                    podSaveFolder += "_d" + str(finiteDelta)
                                elif romSensitivityApproach[isens] == "complex":
                                    podSaveFolder += "_d" + str(complexDelta)
                            else:
                                podSaveFolder = fomSaveFolder
                            if penaltyStrength != 0:
                                podSaveFolder += (
                                    "/"
                                    + mean_reduction[imean]
                                    + "_"
                                    + quadRule[iquad]
                                    + "_n"
                                    + str(nPoints)
                                    + "_p"
                                    + str(penaltyStrength)
                                )
                            else:
                                podSaveFolder += (
                                    "/" + mean_reduction[imean] + "_" + quadRule[iquad] + "_n" + str(nPoints)
                                )

                            if not os.path.exists(podSaveFolder) and plotFullSpectra:
                                os.makedirs(podSaveFolder)

                            # Plot full singular value distribution
                            if iret == 0 and plotFullSpectra:
                                romData, null = model.constructPodRom(
                                    dataModelCoeff[:, :, : 2 * nCollocation * nElements],
                                    x,
                                    W,
                                    min(nT, nPoints),
                                    mean=mean_reduction[imean],
                                    useEnergyThreshold=False,
                                    quadPrecision=True,
                                )
                                fig, axes = plt.subplots(1, 1, figsize=(5, 4))
                                axes.loglog(romData.uSingularValues, "-b", lw=5, ms=8)
                                axes.loglog(romData.vSingularValues, "--m", lw=5, ms=8)
                                axes.legend(["u", "v"])
                                axes.set_xlabel("Mode")
                                axes.set_ylabel("Singular Value")
                                plt.tight_layout()
                                plt.savefig(podSaveFolder + "../singularValues.pdf", format="pdf")
                                plt.savefig(podSaveFolder + "../singularValues.png", format="png")
                            # ------------------------------- Compute POD
                            romData, truncationError[iret] = model.constructPodRom(
                                dataModelCoeff[:, :, : 2 * nCollocation * nElements],
                                x,
                                W,
                                modeRetention[iret],
                                mean=mean_reduction[imean],
                                useEnergyThreshold=useEnergyThreshold,
                                adjustModePairs=False,
                            )
                            romData.penaltyStrength = penaltyStrength

                        error.append(np.empty((neq, len(romParamSamples), len(error_norm))))
                        qoiResults.append(
                            np.empty((len(romParamSamples), len(qois)))
                        )  # Goal: Implement Computation of QoI sensitivity
                        if verbosity >= 1:
                            print(
                                "             Running POD-ROM for mode retention ",
                                modeRetention[iret],
                                " and mean reduction ",
                                mean_reduction[imean],
                            )

                        # Intialize results storage
                        if usePodRom:
                            uResults = np.empty((len(romParamSamples), neq, nT, 3, nPoints))
                            vResults = np.empty((len(romParamSamples), neq, nT, 3, nPoints))
                        else:
                            uResults = np.empty((len(romParamSamples), neq, nT, 1, nPoints))
                            vResults = np.empty((len(romParamSamples), neq, nT, 1, nPoints))
                        combinedResults = np.empty((uResults.shape[0], 2 * neq) + uResults.shape[2:])
                        # Fill initial FOM data
                        uResults[:, :, :, 0, :] = uRefData
                        vResults[:, :, :, 0, :] = vRefData

                        if usePodRom:
                            if nonLinReduction > 1.0 or nDeimPoints != "max":
                                controlSaveFolder = podSaveFolder
                                if nonLinReduction > 1.0:
                                    controlSaveFolder = controlSaveFolder + "nDim" + str(nonLinReduction)
                                if nDeimPoints != "max":
                                    controlSaveFolder = controlSaveFolder + "nDEIM" + str(nDeimPoints)
                                controlSaveFolder = controlSaveFolder + "/"
                            else:
                                controlSaveFolder = podSaveFolder + "noControl/"

                            if useEnergyThreshold:
                                romSaveFolder = controlSaveFolder + "e" + str(modeRetention[iret]) + "/"
                            else:
                                romSaveFolder = controlSaveFolder + "m" + str(modeRetention[iret]) + "/"

                            # Create folders if they don't exist
                            if not os.path.exists(controlSaveFolder) and plotConvergence:
                                os.makedirs(controlSaveFolder)
                            if not os.path.exists(romSaveFolder) and (
                                plotTimeSeries
                                or plotModes
                                or plotError
                                or plotRomCoeff
                                or plotSingularValues
                                or plotRomInterpolation
                                or plotSensitivity
                            ):
                                os.makedirs(romSaveFolder)

                            # Update romData with control approach
                            if nDeimPoints != "max":
                                romData = model.computeDEIMProjection(romData, nDeimPoints)
                            if nonLinReduction > 1.0:
                                romData = model.computeNonLinReduction(
                                    romData, nonLinReduction, proportionality="min singular value"
                                )
                                # Compute time modes for sensitivity equations

                            print(romData.uNonlinDim, ", ", romData.vNonlinDim)

                        # ===== Run POD-ROM =====
                        for iParamSample in range(len(romParamSamples)):
                            if usePodRom:
                                if verbosity >= 2:
                                    print(
                                        "                 Running POD-ROM for parameter sample ",
                                        iParamSample + 1,
                                        " of ",
                                        len(romParamSamples),
                                    )
                                model = model.copy(params=romParamSamples[iParamSample])

                                def dydtPodRom(y, t):
                                    return model.dydtPodRom(y, t, romData, paramSelect=[])

                                # ------------------------------ Run ROM
                                # Get Initial Modal Values
                                # NOTE: Confirm POD indexing and initial condition layout.
                                # All have same initial condition; decide 2D vs 3D array.
                                romInit = np.empty((romData.uNmodes + romData.vNmodes))
                                romInit[: romData.uNmodes] = romData.uTimeModes[0, : romData.uNmodes]
                                romInit[romData.uNmodes : romData.uNmodes + romData.vNmodes] = romData.vTimeModes[
                                    0, : romData.vNmodes
                                ]
                                # Compute Base Rom Value
                                romCoeff = np.empty((nT, neq * (romData.uNmodes + romData.vNmodes)))
                                romCoeff[:, : romData.uNmodes + romData.vNmodes] = model.solve_ivp(
                                    lambda t, y: dydtPodRom(y, t), romInit
                                )

                                # ------------------------------- Compute Sensitivity
                                if equationSet != "tankOnly":
                                    romCoeff = computeSensitivity(
                                        romCoeff,
                                        model,
                                        romData,
                                        paramSelect,
                                        romSensitivityApproach[isens],
                                        sensInit[iInit],
                                        finiteDelta=finiteDelta,
                                        complexDelta=complexDelta,
                                        verbosity=verbosity,
                                    )

                                # ----------------------------- Map Results Back into Spatial Space
                                uResults, vResults = mapROMdataToFOMspace(
                                    romData,
                                    uResults,
                                    vResults,
                                    romCoeff,
                                    iParamSample,
                                    sensInit[iInit],
                                )
                                # ==== Compute Error ====
                                for ieq in range(neq):
                                    for k in range(len(error_norm)):
                                        # Error for ith sample (domain eqs, all times/points): FOM vs ROM
                                        error[iret][ieq, iParamSample, k] = model.computeRomError(
                                            uResults[iParamSample, ieq, :, 0, :].transpose(),
                                            vResults[iParamSample, ieq, :, 0, :].transpose(),
                                            uResults[iParamSample, ieq, :, 1, :].transpose(),
                                            vResults[iParamSample, ieq, :, 1, :].transpose(),
                                            romData.W,
                                            tEval,
                                            norm=error_norm[k],
                                        )
                                        if verbosity >= 2:
                                            print(
                                                "ROM Error in norm " + error_norm[k] + " for ieq " + str(ieq) + ": ",
                                                error[iret][ieq, iParamSample, k],
                                            )
                                # ==== Compute QOIs ====
                                for k in range(len(qois)):
                                    # QOIs for ith sample (domain eqs, all times/points)
                                    qoiResults[iret][iParamSample, k] = model.computeQOIs(
                                        uResults[iParamSample, 0, :, 1, :].transpose(),
                                        vResults[iParamSample, 0, :, 1, :].transpose(),
                                        romData.W,
                                        tEval,
                                        qoi=qois[k],
                                    )
                                    if verbosity >= 2:
                                        print(qois[k] + ": ", qoiResults[iret][iParamSample, k])
                                    # INCOMPLETE: Figure out what want to compute for sensitivity error.
                                for i in range(neq):
                                    combinedResults[iParamSample, 2 * i, :, :, :] = uResults[iParamSample, i, :, :, :]
                                    combinedResults[iParamSample, 2 * i + 1, :, :, :] = vResults[
                                        iParamSample, i, :, :, :
                                    ]
                        # Compute GSA indices
                        if gsaMethod == "DGSM":
                            # Compute time-variant sensitivity indices
                            # Need to re-scale against parameter bounds
                            # Indices are nEq-1 x nT x 3 x nX
                            uDGSMmean = np.sum(np.abs(uResults[:, 1:, :, :, :]), axis=0) / len(romParamSamples)
                            uDGSMvar = np.sum(uResults[:, 1:, :, :, :] ** 2, axis=0) / len(romParamSamples)
                            vDGSMmean = np.sum(np.abs(vResults[:, 1:, :, :, :]), axis=0) / len(romParamSamples)
                            vDGSMvar = np.sum(vResults[:, 1:, :, :, :] ** 2, axis=0) / len(romParamSamples)
                            # Integrate sensitivities in time
                            # Indices are nEq-1 x 3 x nX
                            uDGSMmean = np.sum(uDGSMmean, axis=1) / nT
                            uDGSMvar = np.sum(uDGSMvar, axis=1) / nT
                            vDGSMmean = np.sum(vDGSMmean, axis=1) / nT
                            vDGSMvar = np.sum(vDGSMvar, axis=1) / nT
                        # ======= Plot Results =======
                        if plotSensitivity:
                            figWidth, figHeight = 3.5 * 3, 2.5 * 4
                            fig, axes = plt.subplots(4, 3, figsize=(figWidth, figHeight))
                            plt.subplots_adjust(wspace=0.35)
                            axes = np.reshape(axes, (4, 3))
                            axes[0, 0].semilogy(x, uDGSMmean[:, 0, :].T)
                            axes[0, 0].set_ylabel(r"u, $\mu$")
                            axes[0, 0].set_title("True Sensitivity")
                            axes[0, 1].semilogy(x, uDGSMmean[:, 2, :].T)
                            axes[0, 1].set_title("ROM Sensitivity")
                            axes[0, 2].semilogy(
                                x, np.abs(uDGSMmean[:, 2, :] - uDGSMmean[:, 0, :]).T / np.abs(uDGSMmean[:, 0, :].T)
                            )
                            axes[0, 2].set_title("Relative ROM Error")
                            axes[0, 2].legend(paramSelect)
                            axes[1, 0].semilogy(x, uDGSMvar[:, 0, :].T)
                            axes[1, 0].set_ylabel(r"u, $v$")
                            axes[1, 1].semilogy(x, uDGSMvar[:, 2, :].T)
                            axes[1, 2].semilogy(
                                x, np.abs(uDGSMvar[:, 2, :] - uDGSMvar[:, 0, :]).T / np.abs(uDGSMvar[:, 0, :].T)
                            )
                            axes[2, 0].semilogy(x, vDGSMmean[:, 0, :].T)
                            axes[2, 0].set_ylabel(r"v, $\mu$")
                            axes[2, 1].semilogy(x, vDGSMmean[:, 2, :].T)
                            axes[2, 2].semilogy(
                                x, np.abs(vDGSMmean[:, 2, :] - vDGSMmean[:, 0, :]).T / np.abs(vDGSMmean[:, 0, :].T)
                            )
                            axes[3, 0].semilogy(x, vDGSMvar[:, 0, :].T)
                            axes[3, 0].set_xlabel("x")
                            axes[3, 0].set_ylabel(r"v, $v$")
                            axes[3, 1].semilogy(x, vDGSMvar[:, 2, :].T)
                            axes[3, 1].set_xlabel("x")
                            axes[3, 2].semilogy(
                                x, np.abs(vDGSMvar[:, 2, :] - vDGSMvar[:, 0, :]).T / np.abs(vDGSMvar[:, 0, :].T)
                            )
                            axes[3, 2].set_xlabel("x")
                            plt.tight_layout()
                            plt.savefig(romSaveFolder + "dgsmIndices.pdf", format="pdf")
                            plt.savefig(romSaveFolder + "dgsmIndices.png", format="png")
                        # Plot POD modes
                        if plotModes and usePodRom:
                            subplot(
                                [mode for mode in romData.uModes.transpose()],
                                x,
                                xLabels="x",
                                yLabels=["Mode " + str(i + 1) for i in range(romData.uNmodes)],
                            )
                            plt.savefig(romSaveFolder + "uModes.pdf", format="pdf")
                            plt.savefig(romSaveFolder + "uModes.png", format="png")
                            subplot(
                                [mode for mode in romData.vModes.transpose()],
                                x,
                                xLabels="x",
                                yLabels=["Mode " + str(i + 1) for i in range(romData.vNmodes)],
                            )
                            plt.savefig(romSaveFolder + "vModes.pdf", format="pdf")
                            plt.savefig(romSaveFolder + "vModes.png", format="png")

                            fig, axes = plotRomMatrices(
                                romData.vRomFirstOrderMat,
                                xLabels=r"$\phi'_{v,j}$",
                                yLabels=r"$\phi_{v,i}$",
                                title=r"1st Order v ROM matrix: $\langle\phi_{v,i},\phi'_{v,j}\rangle_{L^2}$",
                                cmap="coolwarm",
                            )
                            plt.savefig(romSaveFolder + "vRomMatrices_1stOrder.pdf", format="pdf")
                            plt.savefig(romSaveFolder + "vRomMatrices_1stOrder.png", format="png")
                            fig, axes = plotRomMatrices(
                                romData.vRomSecondOrderMat,
                                xLabels=r"$\phi''_{v,j}$",
                                yLabels=r"$\phi_{v,i}$",
                                title=r"2nd Order v ROM matrix: $\langle\phi_{v,i},\phi''_{v,j}\rangle_{L^2}$",
                                cmap="coolwarm",
                            )
                            plt.savefig(romSaveFolder + "vRomMatrices_2ndOrder.pdf", format="pdf")
                            plt.savefig(romSaveFolder + "vRomMatrices_2ndOrder.png", format="png")
                            fig, axes = plotRomMatrices(
                                [romData.vRomFirstOrderMat, romData.vRomSecondOrderMat],
                                xLabels=[r"$\phi'_{v,j}$", r"$\phi''_{v,j}$"],
                                yLabels=r"$\phi_{v,i}$",
                                title=[
                                    r"1st Order v ROM matrix: $\langle\phi_{v,i},\phi'_{v,j}\rangle_{L^2}$",
                                    r"2nd Order v ROM matrix: $\langle\phi_{v,i},\phi''_{v,j}\rangle_{L^2}$",
                                ],
                                cmap="coolwarm",
                            )
                            plt.savefig(romSaveFolder + "vRomMatrices.pdf", format="pdf")
                            plt.savefig(romSaveFolder + "vRomMatrices.png", format="png")

                            fig, axes = plotRomMatrices(
                                romData.uRomFirstOrderMat,
                                xLabels=r"$\phi'_{u,j}$",
                                yLabels=r"$\phi_{u,i}$",
                                title=r"1st Order u ROM matrix: $\langle\phi_{u,i},\phi'_{u,j}\rangle_{L^2}$",
                                cmap="coolwarm",
                            )
                            plt.savefig(romSaveFolder + "uRomMatrices_1stOrder.pdf", format="pdf")
                            plt.savefig(romSaveFolder + "uRomMatrices_1stOrder.png", format="png")
                            fig, axes = plotRomMatrices(
                                romData.uRomSecondOrderMat,
                                xLabels=r"$\phi''_{u,j}$",
                                yLabels=r"$\phi_{u,i}$",
                                title=r"2nd Order u ROM matrix: $\langle\phi_{u,i},\phi''_{u,j}\rangle_{L^2}$",
                                cmap="coolwarm",
                            )
                            plt.savefig(romSaveFolder + "uRomMatrices_2ndOrder.pdf", format="pdf")
                            plt.savefig(romSaveFolder + "uRomMatrices_2ndOrder.png", format="png")
                            fig, axes = plotRomMatrices(
                                [romData.uRomFirstOrderMat, romData.uRomSecondOrderMat],
                                xLabels=[r"$\phi'_{u,j}$", r"$\phi''_{u,j}$"],
                                yLabels=r"$\phi_{u,i}$",
                                title=[
                                    r"1st Order u ROM matrix: $\langle\phi_{u,i},\phi'_{u,j}\rangle_{L^2}$",
                                    r"2nd Order u ROM matrix: $\langle\phi_{u,i},\phi''_{u,j}\rangle_{L^2}$",
                                ],
                                cmap="coolwarm",
                            )
                            plt.savefig(romSaveFolder + "uRomMatrices.pdf", format="pdf")
                            plt.savefig(romSaveFolder + "uRomMatrices.png", format="png")

                        legends = ["FOM", "POD", "ROM"]
                        if plotRomCoeff:
                            coeffLabels = ["Coeff " + str(i + 1) for i in range(romData.uNmodes)]
                            uCoeffData = [
                                np.array([rom, pod])
                                for pod, rom in zip(
                                    romData.uTimßeModes[:, : romData.uNmodes].transpose(),
                                    romCoeff[:, : romData.uNmodes].transpose(),
                                )
                            ]

                            fig, axs = subplot(
                                uCoeffData,
                                tEval,
                                xLabels="t",
                                yLabels=coeffLabels,
                                legends=legends[1:3],
                                subplotSize=(2.65, 2),
                                lineTypeStart=1,
                            )
                            plt.savefig(romSaveFolder + "uRomCoeff.pdf", format="pdf")
                            plt.savefig(romSaveFolder + "uRomCoeff.png", format="png")

                            coeffLabels = ["Coeff " + str(i + 1) for i in range(romData.vNmodes)]
                            vCoeffData = [
                                np.array([rom, pod])
                                for pod, rom in zip(
                                    romData.vTimeModes[:, : romData.vNmodes].transpose(),
                                    romCoeff[:, romData.uNmodes : romData.uNmodes + romData.vNmodes].transpose(),
                                )
                            ]
                            fig, axs = subplot(
                                vCoeffData,
                                tEval,
                                xLabels="t",
                                yLabels=coeffLabels,
                                legends=legends[1:3],
                                subplotSize=(2.65, 2),
                                lineTypeStart=1,
                            )
                            plt.savefig(romSaveFolder + "vRomCoeff.pdf", format="pdf")
                            plt.savefig(romSaveFolder + "vRomCoeff.png", format="png")
                        # Plot singular values
                        if plotSingularValues:
                            # Compute culmulative truncation
                            nSV = max(romData.uSingularValues.size, romData.vSingularValues.size)
                            totalTruncation = 1 - np.cumsum(
                                np.pad(romData.uSingularValues, (0, nSV - romData.uSingularValues.size))
                                + np.pad(romData.vSingularValues, (0, nSV - romData.vSingularValues.size))
                            ) / np.sum(romData.uFullSpectra + romData.vFullSpectra)
                            # uCutoffIndices = np.array([romData.uSingularValues.size-np.sum(uPropInformation<=.1),
                            #                            romData.uSingularValues.size-np.sum(uPropInformation<=.01),
                            #                            romData.uSingularValues.size-np.sum(uPropInformation<=.001),
                            #                            romData.uSingularValues.size-np.sum(uPropInformation<=.0001)])
                            # vCutoffIndices = np.array([romData.vSingularValues.size-np.sum(vPropInformation<=.1),
                            #                            romData.vSingularValues.size-np.sum(vPropInformation<=.01),
                            #                            romData.vSingularValues.size-np.sum(vPropInformation<=.001),
                            #                            romData.vSingularValues.size-np.sum(vPropInformation<=.0001)])
                            cutoffIndices = np.array(
                                [
                                    nSV - np.sum(totalTruncation <= 0.1),
                                    nSV - np.sum(totalTruncation <= 0.01),
                                    nSV - np.sum(totalTruncation <= 0.001),
                                    nSV - np.sum(totalTruncation <= 0.0001),
                                ]
                            )
                            cutoffIndices = cutoffIndices[
                                (cutoffIndices < romData.uSingularValues.size)
                                & (cutoffIndices < romData.vSingularValues.size)
                            ]
                            uCutoffSv = romData.uSingularValues[cutoffIndices]
                            vCutoffSv = romData.vSingularValues[cutoffIndices]
                            padding = 1.3
                            width = 0.95
                            cutoffSV_padded = np.array(
                                [
                                    np.maximum(uCutoffSv, vCutoffSv) * padding,
                                    np.minimum(uCutoffSv, vCutoffSv) / padding,
                                ]
                            ).T
                            fig, axes = plt.subplots(1, 1, figsize=(5, 4))
                            axes.semilogy(
                                np.arange(1, romData.uSingularValues.size + 1),
                                romData.uSingularValues,
                                "bs",
                                lw=5,
                                ms=5,
                            )
                            axes.semilogy(
                                np.arange(1, romData.vSingularValues.size + 1),
                                romData.vSingularValues,
                                "mo",
                                lw=5,
                                ms=5,
                            )
                            labels = ["90%", "99%", "99.9%", "99.99%"]
                            for i in range(cutoffIndices.size):
                                rect = Rectangle(
                                    (cutoffIndices[i] + 1 - width / 2, cutoffSV_padded[i, 1]),
                                    width,
                                    cutoffSV_padded[i, 0] - cutoffSV_padded[i, 1],
                                    facecolor="none",
                                    edgecolor="k",
                                    linewidth=2,
                                    zorder=3,
                                )
                                plt.gca().add_patch(rect)
                                # Place label just above the TOP edge of the rectangle (robust for log-scale)
                                y_top = cutoffSV_padded[i, 0]
                                x_right = cutoffIndices[i] + 1 + width / 2
                                # Use a small offset so it always appears
                                # above the line regardless of scale
                                axes.annotate(
                                    labels[i],
                                    xy=(x_right, y_top),
                                    xytext=(0, 4),
                                    textcoords="offset points",
                                    ha="center",
                                    va="bottom",
                                    fontsize=10,
                                    color="k",
                                    zorder=5,
                                )

                            # Create legend for first and third lines (indices 0 and 2)
                            plt.legend(["u", "v"])

                            axes.set_xlabel("Mode")
                            axes.set_ylabel("Singular Value")
                            plt.tight_layout()
                            plt.savefig(romSaveFolder + "singularValues.pdf", format="pdf")
                            plt.savefig(romSaveFolder + "singularValues.png", format="png")

                            fig, axes = plt.subplots(1, 1, figsize=(5, 4))
                            uPropInformation = 1 - np.cumsum(romData.uSingularValues) / np.sum(romData.uFullSpectra)
                            vPropInformation = 1 - np.cumsum(romData.vSingularValues) / np.sum(romData.vFullSpectra)
                            axes.semilogy(
                                np.arange(1, romData.uSingularValues.size + 1),
                                uPropInformation[: romData.uSingularValues.size],
                                "bs",
                                lw=5,
                                ms=8,
                            )
                            axes.semilogy(
                                np.arange(1, romData.vSingularValues.size + 1),
                                vPropInformation[: romData.vSingularValues.size],
                                "mo",
                                lw=5,
                                ms=8,
                            )
                            axes.legend(["u", "v"])
                            axes.set_xlabel("Mode")
                            axes.set_ylabel("Singular Value")
                            plt.tight_layout()
                            plt.savefig(romSaveFolder + "propInformation.pdf", format="pdf")
                            plt.savefig(romSaveFolder + "propInformation.png", format="png")
                        if not showPlots:
                            plt.close()
                    # Plot convergence
                    if usePodRom and plotConvergence and len(error) > 1:
                        # Don't plot error convergence for sensitivities or multiple parameter values
                        errorPlot = np.array([errorRet[0, 0, :] for errorRet in error]).T.tolist()
                        errorPlot = [np.array(error) for error in errorPlot]
                        # Note: plotErrorConvergnece was refactored for
                        # lists of errors; this indexing may need revisiting
                        if len(mean_reduction) > 1:
                            legends = [
                                mean_method + ", " + norm for mean_method in mean_reduction for norm in error_norm
                            ]
                        else:
                            legends = error_norm
                        fig, axs = plotErrorConvergence(
                            errorPlot,
                            truncationError,
                            xLabel="Proportion Information Truncated in POD",
                            yLabel="Relative ROM Error",
                            legends=legends,
                        )
                        plt.savefig(
                            controlSaveFolder
                            + "errorConvergence_s"
                            + str(modeRetention[0])
                            + "_e"
                            + str(modeRetention[-1])
                            + ".pdf",
                            format="pdf",
                        )
                        plt.savefig(
                            controlSaveFolder
                            + "errorConvergence_s"
                            + str(modeRetention[0])
                            + "_e"
                            + str(modeRetention[-1])
                            + ".png",
                            format="png",
                        )

    if showPlots:
        plt.show()


if __name__ == "__main__":
    main()
