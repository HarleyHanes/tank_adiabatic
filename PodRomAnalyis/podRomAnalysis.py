#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   podRomAnalysis/podRomAnalysis.py
@Time    :   2025/11/18 11:16:26
@Author  :   Harley Hanes 
@Version :   1.0
@Contact :   hhanes@ncsu.edu
@License :   (C)Copyright 2025, Harley Hanes
@Desc    :   Support functions for POD-ROM analysis
'''

import numpy as np

def computeControlMetric(error,controlParam,truncationError,metric):
    if metric == "Min Error":
        metricResult = np.min(error,axis=0)
    elif metric == "Mean Error":
        metricResult = np.mean(error,axis=0)
        if np.any(truncationError < .0001):
            index = np.argmax(truncationError < .0001)
        else:
            index=-1
        metricResult = error[index,:]
    elif metric == "Max Error Increase":
        metricResult = np.max((error[1:]-error[:-1])/error[:-1])
    elif metric == "Sum of Relative Error Increases":
        errorChanges = (error[1:]-error[:-1])/error[:-1]
        errorChanges[errorChanges<0]=0
        metricResult = np.nansum(errorChanges,axis=0)
    elif metric == "Number Error Increases":
        errorChanges = (error[1:]-error[:-1])/error[:-1]
        errorChanges[errorChanges<0]=0
        errorChanges[errorChanges>0]=1
        metricResult = np.nansum(errorChanges,axis=0)
    else:
        raise(ValueError("Invalid metric entered: ", metric))
    return metricResult[::-1]

def computeInitialCondition(model, neq):
    period = 1
    init = lambda x,b: b[0]+b[1]/model.bounds[1]*x+b[2]*np.cos(2*np.pi*x*period/model.bounds[1])+b[3]*np.sin(2*np.pi*x*period/model.bounds[1])
    uCoeff = np.empty((4,))
    uCoeff[0]=0
    uCoeff[1]=1
    uCoeff[2]=-uCoeff[0]
    uCoeff[3]=-uCoeff[1]/(2*np.pi*period)

    vCoeff = np.empty((4,))
    vCoeff[0]=.35
    vCoeff[1]=.2
    vCoeff[2]=-vCoeff[0]+(model.params["f"]*vCoeff[1])/(1-model.params["f"])
    vCoeff[3]=-vCoeff[1]/(2*np.pi*period)
    modelCoeff=np.append(init(model.collocationPoints,uCoeff),init(model.collocationPoints,vCoeff),axis=0)
    #Deprecating sensitivity initial conditions for now
    # for i in range(neq-1):
    #     modelCoeff = np.append(modelCoeff,0*model.collocationPoints,axis=0)
    #     modelCoeff = np.append(modelCoeff,0*model.collocationPoints,axis=0) #Changed to 0 initial condition for both u and v

    return modelCoeff

def getSensitivityOptions(equationSet):
    if equationSet == "tankOnly":
        neq=1
        paramSelect=[]
        uLabels=[r"$u$"]
        vLabels=[r"$v$"]
        combinedLabels= [r"$u$",r"$v$"]
    elif equationSet == "Le":
        neq=2
        paramSelect=["Le"]
        uLabels=[r"$u$",r"$u_{\mathrm{Le}}$"]
        vLabels=[r"$v$",r"$v_{\mathrm{Le}}$"]
        combinedLabels= [r"$u$",r"$v$",r"$u_{\mathrm{Le}}$",r"$v_{\mathrm{Le}}$"]
    elif equationSet == "vH":
        neq=2
        paramSelect=["vH"]
        uLabels=[r"$u$",r"$u_{v_H}$"]
        vLabels=[r"$v$",r"$v_{v_H}$"]
        combinedLabels= [r"$u$",r"$v$",r"$u_{v_H}$",r"$v_{v_H}$"]
    elif equationSet == "gamma":
        neq=2
        paramSelect=["gamma"]
        uLabels=[r"$u$",r"$u_{\gamma}$"]
        vLabels=[r"$v$",r"$v_{\gamma}$"]
        combinedLabels= [r"$u$",r"$v$",r"$u_{\gamma}$",r"$v_{\gamma}$"]
    elif equationSet == "linearParams":
        neq=4
        paramSelect=["Le","delta","vH"]
        uLabels=[r"$u$",r"$u_{\mathrm{Le}}$",r"$u_{\delta}$",r"$u_{v_H}$"]
        vLabels=[r"$v$",r"$v_{\mathrm{Le}}$",r"$v_{\delta}$",r"$v_{v_H}$"]
        combinedLabels= [r"$u$",r"$u_{\mathrm{Le}}$",r"$u_{\delta}$",r"$u_{v_H}$",r"$v$",r"$v_{\mathrm{Le}}$",r"$v_{\delta}$",r"$v_{v_H}$"]
    elif equationSet == "linearBoundaryParams":
        neq=8
        paramSelect=["PeM","PeT","f","Le","Da","delta","vH"]
        uLabels=[r"$u$",r"$u_{\mathrm{Pe_M}}$",r"$u_{\mathrm{Pe_T}}$",r"$u_{f}$",r"$u_{\mathrm{Le}}$",r"$u_{\mathrm{Da}}$",r"$u_{\delta}$",r"$u_{v_H}$"]
        vLabels=[r"$v$",r"$v_{\mathrm{Pe_M}}$",r"$v_{\mathrm{Pe_T}}$",r"$v_{f}$",r"$v_{\mathrm{Le}}$",r"$v_{\mathrm{Da}}$",r"$v_{\delta}$",r"$v_{v_H}$"]
        combinedLabels= [r"$u$",r"$u_{\mathrm{Pe_M}}$",r"$u_{\mathrm{Pe_T}}$",r"$u_{f}$",r"$u_{\mathrm{Le}}$",r"$u_{\mathrm{Da}}$",r"$u_{\delta}$",r"$u_{v_H}$",r"$v$",r"$v_{\mathrm{Pe_M}}$",r"$v_{\mathrm{Pe_T}}$",r"$v_{f}$",r"$v_{\mathrm{Le}}$",r"$v_{\mathrm{Da}}$",r"$v_{\delta}$",r"$v_{v_H}$"]
    elif equationSet == "allParams":
        neq=10
        paramSelect=["PeM","PeT","f","Le","Da","beta","gamma","delta","vH"]
        uLabels=[r"$u$",r"$u_{\mathrm{Pe_M}}$",r"$u_{\mathrm{Pe_T}}$",r"$u_{f}$",r"$u_{\mathrm{Le}}$",r"$u_{\mathrm{Da}}$",r"$u_{\beta}$",r"$u_{\gamma}$",r"$u_{\delta}$",r"$u_{v_H}$"]
        vLabels=[r"$v$",r"$v_{\mathrm{Pe_M}}$",r"$v_{\mathrm{Pe_T}}$",r"$v_{f}$",r"$v_{\mathrm{Le}}$",r"$v_{\mathrm{Da}}$",r"$v_{\beta}$",r"$v_{\gamma}$",r"$v_{\delta}$",r"$v_{v_H}$"]
    elif equationSet == "nonBoundaryParams":
        neq=7
        paramSelect=["Le","Da","beta","gamma","delta","vH"]
        uLabels=[r"$u$",r"$u_{\mathrm{Le}}$",r"$u_{\mathrm{Da}}$",r"$u_{\beta}$",r"$u_{\gamma}$",r"$u_{\delta}$",r"$u_{v_H}$"]
        vLabels=[r"$v$",r"$v_{\mathrm{Le}}$",r"$v_{\mathrm{Da}}$",r"$v_{\beta}$",r"$v_{\gamma}$",r"$v_{\delta}$",r"$v_{v_H}$"]
        combinedLabels=[r"$u$",r"$v$",r"$u_{\mathrm{Le}}$",r"$v_{\mathrm{Le}}$",r"$u_{\mathrm{Da}}$",r"$v_{\mathrm{Da}}$",r"$u_{\beta}$",r"$v_{\beta}$",r"$u_{\gamma}$",r"$v_{\gamma}$",r"$u_{\delta}$",r"$v_{\delta}$",r"$u_{v_H}$",r"$v_{v_H}$"]
    else:
        raise ValueError("Invalid equationSet entered: " + str(equationSet))
    return neq, paramSelect, uLabels, vLabels, combinedLabels

def getParameterOptions(paramSet):
    stabalized=False #Default No Stabalization
    if paramSet == "BizonChaotic":
        baseParams={"PeM": 700, "PeT": 700, "f": .3, "Le": 1, "Da": .15, "beta": 1.8, "gamma": 10,"delta": 2, "vH":-.065}
        stabalized=True
    elif paramSet == "BizonPeriodic":
        baseParams={"PeM": 300, "PeT": 300, "f": .3, "Le": 1, "Da": .15, "beta": 1.4, "gamma": 10,"delta": 2, "vH":-.045}
        stabalized=True 
    elif paramSet == "BizonPeriodicReduced":
        baseParams={"PeM": 300, "PeT": 300, "f": .3, "Le": 1, "Da": .08966443, "beta": 1.4, "gamma": 10,"delta": 2, "vH":-.045}
    elif paramSet == "BizonLinear":
        baseParams={"PeM": 700, "PeT": 700, "f": .3, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 2, "vH":-.065}
    elif paramSet == "BizonNonLinear":
        baseParams={"PeM": 700, "PeT": 700, "f": .3, "Le": 1, "Da": .15, "beta": 1.8, "gamma": 10,"delta": 2, "vH":-.065}
    elif paramSet == "BizonLinearNoRobin":
        baseParams={"PeM": 300, "PeT": 300, "PeM-boundary": 1e16, "PeT-boundary": 1e16, "f": .3, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 2, "vH":-.045}
    elif paramSet == "BizonAdvecDiffusion":
        baseParams={"PeM": 300, "PeT": 300, "f": .3, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 0, "vH":0}
    elif paramSet == "BizonAdvecDiffusionNoRobin":
        baseParams={"PeM": 300, "PeT": 300, "PeM-boundary": 1e16, "PeT-boundary": 1e16, "f": .3, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 0, "vH":0}
    elif paramSet == "BizonAdvecDiffusionNoRobinNoRecirc":
        baseParams={"PeM": 300, "PeT": 300, "PeM-boundary": 1e16, "PeT-boundary": 1e16, "f": 0, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 0, "vH":0}
    elif paramSet == "NoRecircExtreme":
        baseParams={"PeM": 300, "PeT": 300, "f": 0, "Le": 1, "Da": .5, "beta": 1.4, "gamma": 10,"delta": 2, "vH":-.045}
    elif paramSet == "NoRecirc":
        baseParams={"PeM": 1e2, "PeT": 1e2, "f": 0, "Le": 1, "Da": .15, "beta": 1.4, "gamma": 10,"delta": 2, "vH":-.045}
    elif paramSet == "AdvecDiffusionNonLinear":
        baseParams={"PeM": 1e2, "PeT": 1e2, "f": 0, "Le": 1, "Da": .15, "beta": 1.5, "gamma": 10,"delta": 0, "vH":0}
    elif paramSet == "AdvecNonLinear":
        baseParams={"PeM": 1e16, "PeT": 1e16, "f": 0, "Le": 1, "Da": .15, "beta": 0, "gamma": 0,"delta": 0, "vH":0}
    elif paramSet == "AdvecDiffusionLinearRecirc":
        baseParams={"PeM": 1e2, "PeT": 1e2, "f": .75, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 2, "vH":-.045}
    elif paramSet == "AdvecDiffusionLinear":
        baseParams={"PeM": 1e2, "PeT": 1e2, "f": 0, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 2, "vH":-.045}
    elif paramSet == "AdvecLinearRecirc":
        baseParams={"PeM": 1e16, "PeT": 1e16, "f": .75, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 2, "vH":-.045}
    elif paramSet == "AdvecLinear":
        baseParams={"PeM": 1e16, "PeT": 1e16, "f": 0, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 2, "vH":-.045}
    elif paramSet == "AdvecDiffusionRecircExtreme":
        baseParams={"PeM": 1e2, "PeT": 1e2, "f": 4, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 0, "vH":0}
    elif paramSet == "AdvecDiffusionRecircExtremeNoRobin":
        baseParams={"PeM": 1e2, "PeT": 1e2, "PeM-boundary": 1e16, "PeT-boundary": 1e16, "f": 4, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 0, "vH":0}
    elif paramSet == "AdvecDiffusionRecirc":
        baseParams={"PeM": 1e2, "PeT": 1e2, "f": 1, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 0, "vH":0}
    elif paramSet == "AdvecDiffusionRecircNoRobin":
        baseParams={"PeM": 1e2, "PeT": 1e2,"PeM-boundary": 1e16, "PeT-boundary": 1e16, "f": 1, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 0, "vH":0}
    elif paramSet == "AdvecRecirc":
        baseParams={"PeM": 1e16, "PeT": 1e16, "f": .75, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 0, "vH":0}
    elif paramSet == "AdvecDiffusion":
        baseParams={"PeM": 1e2, "PeT": 1e2, "f": 0, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 0, "vH":0}
    elif paramSet == "Advec":
        baseParams={"PeM": 1e16, "PeT": 1e16, "f": 0, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 0, "vH":0}
    else: 
        raise ValueError("Invalid paramSet entered: " + str(paramSet))
    return baseParams,stabalized

def computeSensitivity(romCoeff,model,romData,paramSelect,romSensitivityApproach,sensInit,finiteDelta=1e-8, complexDelta=1e-14,verbosity =0):
    for iparam in range(len(paramSelect)):
        if romSensitivityApproach == "finite":
            if verbosity >= 1:
                print("Computing sensitivity for " + paramSelect[iparam])
            if sensInit=="pod":
                raise Exception("POD initialization not currently supported for finite-difference sensitivity")
                # #If finite-diff is exact at t=0 then u(0,x+delta)=u(0,x)+delta*u'(0,x)
                # perturbedRomCoeff = romCoeff[0,:romData.uNmodes+romData.vNmodes]
                # perturbedRomCoeff[:romData.uNmodes] +=\
                #     finiteDelta*uFullTimeModes[0,(iparam+1)*romData.uNmodes:(iparam+2)*romData.uNmodes]
                # perturbedRomCoeff[romData.uNmodes:romData.uNmodes+romData.vNmodes] += \
                #     finiteDelta*vFullTimeModes[0,(iparam+1)*romData.vNmodes:(iparam+2)*romData.vNmodes]
            elif sensInit=="zero":
                #If finite-diff is zero at t=0 then u(0,x+delta)=u(0,x)
                perturbedRomCoeff = romCoeff[0,:romData.uNmodes+romData.vNmodes]
            perturbedParams = model.params.copy()
            perturbedParams[paramSelect[iparam]]+= finiteDelta
            #NOTE: Should add other discretization details to model: tmax, tEval,odeMethod, penaltyStrength
            perturbedModel = model.copy(params=perturbedParams)
            dydtPodRom = lambda y,t: perturbedModel.dydtPodRom(y,t,romData,paramSelect = [])
            #Compute Sensitivity in POD space
            romCoeff[:,(iparam+1)*(romData.uNmodes+romData.vNmodes):(iparam+2)*(romData.uNmodes+romData.vNmodes)]=\
                (model.solve_ivp(lambda t,y: dydtPodRom(y,t),perturbedRomCoeff)-romCoeff[:,:romData.uNmodes+romData.vNmodes])/finiteDelta
        elif romSensitivityApproach == "complex":
            if verbosity >= 1:
                print("Computing sensitivity for " + paramSelect[iparam])
            #Initialize sensitivity
            if sensInit=="pod":
                raise Exception("POD initialization not currently supported for complex-step sensitivity")
                # #If finite-diff is exact at t=0 then u(0,x+delta)=u(0,x)+delta*u'(0,x)
                # perturbedRomCoeff = romCoeff[0,:romData.uNmodes+romData.vNmodes].astype(complex)
                # perturbedRomCoeff[:romData.uNmodes] += \
                #     1j*complexDelta*uFullTimeModes[0,(iparam+1)*romData.uNmodes:(iparam+2)*romData.uNmodes]
                # perturbedRomCoeff[romData.uNmodes:romData.uNmodes+romData.vNmodes] += \
                #     1j*complexDelta*vFullTimeModes[0,(iparam+1)*romData.vNmodes:(iparam+2)*romData.vNmodes]
            elif sensInit=="zero":
                #If finite-diff is zero at t=0 then u(0,x+delta)=u(0,x)
                perturbedRomCoeff = romCoeff[0,:romData.uNmodes+romData.vNmodes].astype(complex)
            perturbedParams = model.params.copy()
            perturbedParams[paramSelect[iparam]] += complexDelta*1j
            perturbedModel = model.copy(params=perturbedParams)
            dydtPodRom = lambda y,t: perturbedModel.dydtPodRom(y,t,romData,paramSelect = [])
            romCoeff[:,(iparam+1)*(romData.uNmodes+romData.vNmodes):(iparam+2)*(romData.uNmodes+romData.vNmodes)]=\
                  np.imag(model.solve_ivp(lambda t,y: dydtPodRom(y,t),perturbedRomCoeff))/complexDelta
        elif romSensitivityApproach == "sensEq":
            if verbosity >= 1:
                print("Computing sensitivity for " + paramSelect[iparam])
            romInit = np.empty((2*(romData.uNmodes+romData.vNmodes)))
            romInit[:romData.uNmodes+romData.vNmodes] = romCoeff[0,:romData.uNmodes+romData.vNmodes].copy() 
            if sensInit=="pod":
                raise Exception("POD initialization not currently supported for sensitivity equation approach")
                # romInit[romData.uNmodes+romData.vNmodes:2*romData.uNmodes+romData.vNmodes]\
                #     =uFullTimeModes[0,(iparam+1)*romData.uNmodes:(iparam+2)*romData.uNmodes]
                # romInit[2*romData.uNmodes+romData.vNmodes:]\
                #     =vFullTimeModes[0,(iparam+1)*romData.vNmodes:(iparam+2)*romData.vNmodes]
            elif sensInit=="zero":
                romInit[romData.uNmodes+romData.vNmodes:] =np.zeros((romData.uNmodes+romData.vNmodes))
            else:
                raise ValueError("Invalid sensInit entered: " + str(sensInit))
            dydtPodRom = lambda y,t: model.dydtPodRom(y,t,romData,paramSelect = paramSelect[iparam])

            #Compute Sensitivity in POD space
            romCoeff[:,(iparam+1)*(romData.uNmodes+romData.vNmodes):(iparam+2)*(romData.uNmodes+romData.vNmodes)]=\
                model.solve_ivp(lambda t,y: dydtPodRom(y,t), romInit)[:,romData.uNmodes+romData.vNmodes:]
    return romCoeff


def constructParameterSamples(baseParams, paramBounding,param,nRomSamples):
    if param == "none":
        # No parameter sampling
        fomParamSamples = [baseParams]
        romParamSamples = [baseParams]
    else: 
        base_val = baseParams[param]
        lo = base_val * (1 - paramBounding)
        hi = base_val * (1 + paramBounding)

        # FOM: 3 samples (lo, base, hi)
        fom_values = np.linspace(lo, hi, 3).tolist()
        fomParamSamples = [{**baseParams, param: v} for v in fom_values]

        # ROM: nRomSamples across the same interval
        rom_values = np.linspace(lo, hi, nRomSamples).tolist()
        romParamSamples = [{**baseParams, param: v} for v in rom_values]

    return fomParamSamples, romParamSamples