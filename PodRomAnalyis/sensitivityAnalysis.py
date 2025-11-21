from cmath import rect
import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath(os.path.join(current_script_dir,'..',))
sys.path.append(grandparent_dir)
import numpy as np
import scipy
from postProcessing.plot import subplotTimeSeries
from postProcessing.plot import subplot
from postProcessing.plot import subplotMovie
from postProcessing.plot import plotErrorConvergence
from postProcessing.plot import plotRomMatrices
from podRomAnalysis import computeControlMetric
from podRomAnalysis import computeInitialCondition
from podRomAnalysis import getSensitivityOptions
from podRomAnalysis import getParameterOptions
from podRomAnalysis import computeSensitivity
from podRomAnalysis import constructParameterSamples
from tankModel.TankModel import TankModel
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def main():
    #================================================================Define Simulation Details===============================================================================================
    verbosity =1
    showPlots= True
    #Run Types
    plotConvergence=False

    plotRomInterpolation = True

    plotTimeSeries=True
    plotModes=False
    plotError=True
    plotRomCoeff=False
    plotSingularValues=False 
    plotFullSpectra = False

    makeMovies=False
    #FOM parameters
    paramSet = "BizonChaotic" #BizonPeriodic, BizonLinear, BizonChaotic, BizonAdvecDiffusion
    equationSet = "tankOnly" #tankOnly, Le, vH, linearParams, linearBoundaryParams, allParams, nonBoundaryParams
    nCollocation=2
    nElements=64
    odeMethod="BDF" #LSODA, BDF, Note: Need a stiff solver, LSODA fastest but BDF needed to support complex step
    nPoints=599
    nT=600

    #Parameter Sampling
    param ="vH"
    if param != "none":
        extrapolatory = False
        equationSet = param #Comment out to do parameter sampling without sensitivity
        paramBounding = .5
        nRomSamples = 3

    #ROM parameters
    usePodRom=True
    useEnergyThreshold=True
    nDeimPoints = "max" #Base value for DEIM, max or integer
    nonLinReduction = 4.0 #Base value for nonLinReduction, 1 means no reduction
    penaltyStrength=0
    sensInit = ["zero"]
    quadRule = ["gauss-legendre"] # simpson, gauss-legendre, uniform, monte carlo
    mean_reduction = ["mean"]
    error_norm = [r"$L_2$ Error",r"$L_\infty$ Error"]
    qois = ["Max Reactivity", "Average Reactivity", "Max Outlet Temperature", "Average Outlet Temperature"]
    romSensitivityApproach = ["sensEq","complex"] #none, finite, sensEq, complex, only used if equationSet!=tankOnly
    finiteDelta = 1e-6   #Only used if equationSet!=tankOnly and romSensitivityApproach=="finite"
    complexDelta = 1e-9 #Only used if equationSet!=tankOnly and romSensitivityApproach=="complex"

    #================================================================= Set simulation parameters ==============================================================================================
    #Set POD Retention
    if useEnergyThreshold==True:
        modeRetention=.995
    else:
        if plotConvergence:
            if paramSet=="BizonChaotic":
                modeRetention = list(range(6,53)) #1e-1 to 1e-5
                #modeRetention = list(range(6,39)) #1e-1 to 1e-4
                #modeRetention = list(range(6,89))
            elif paramSet=="BizonPeriodic":
                modeRetention = list(range(5,32)) #1e-1 to 1e-5
                #modeRetention = list(range(6,29))
            elif paramSet=="BizonLinear":
                #modeRetention = list(range(1,15))
                #modeRetention = list(range(2,15)) #1e-1 to 1e-5
                modeRetention = list(range(2,20)) #1e-1 to 1e-5
            elif paramSet== "BizonNonLinear":
                modeRetention = list(range(7,59))
        else:
            modeRetention = [27]


    #Change all POD-ROM parameters to lists if not  already
    if type(modeRetention)==float or type(modeRetention)==int:
        modeRetention = [modeRetention]
    if type(mean_reduction)==str:
        mean_reduction = [mean_reduction]
    if equationSet != "tankOnly":
        if type(romSensitivityApproach)==str:
            romSensitivityApproach = [romSensitivityApproach]
        if type(sensInit)==str:
            sensInit = [sensInit]
    else : 
        romSensitivityApproach = ["none"]
        sensInit = ["none"]
    if type(quadRule)==str:
        quadRule = [quadRule]

    #Get Parameter Settings
    baseParams, stabalized = getParameterOptions(paramSet)

    fomSaveFolder = "../../results/sensitivityAnalysis/"+paramSet
    
    fomSaveFolder += "_nCol" + str(nCollocation) + "_nElem"+str(nElements) + "_nT" + str(nT) +"_nX"+str(nPoints)+"_"+odeMethod+"/"+equationSet
    if not os.path.exists(fomSaveFolder):
        os.makedirs(fomSaveFolder)      
    #Simulation Settings
    bounds = [0,1]
    if stabalized:
        stabalizationTime=150
        tmax=4.1
    else:
        tmax=1.5
    tEval= np.linspace(0,tmax,num=nT)
    #Determine parameters to get sensitivity of
    neq, paramSelect, uLabels, vLabels, combinedLabels = getSensitivityOptions(equationSet)
    #=================================== Construct Parameter Samples =========================================================================
    # Determine parameter samples (lower, center, upper for FOM; evenly spaced for ROM)
    fomParamSamples, romParamSamples = constructParameterSamples(baseParams, paramBounding,param,nRomSamples,extrapolatory)
    

    #==================================== Setup system ===============================================================================
    if verbosity >= 1:
        print("Setting up system")
    model = TankModel(nCollocation=nCollocation,nElements=nElements,spacing="legendre",bounds=bounds,params=baseParams, 
                      tEval = tEval, odeMethod = odeMethod, penaltyStrength = penaltyStrength, verbosity=verbosity)
    dydtSens =lambda y,t: model.dydtSens(y,t,paramSelect=paramSelect)
    dydtStabalization =lambda y,t: model.dydtSens(y,t,paramSelect=[])


    #=================================== Run Stabalization ===========================================================================
    if verbosity >= 1:
        print("Running Stabalization")
    if stabalized:
        #Run out till stabalizing in periodic domain
        initialCondition = np.zeros(model.nCollocation*model.nElements*2*neq)
        #Note: Sensitivities have 0 initial condition
        initialCondition[:model.nCollocation*model.nElements*2] = model.solve_ivp(lambda t,y: dydtStabalization(y,t), initialCondition[:model.nCollocation*model.nElements*2],\
                                                                                  tEval = [0,stabalizationTime])[-1,:]
    else:
        initialCondition = computeInitialCondition(model,neq)
    #=================================== Get Simulation Data ================================================================
    #Step 1: Get FOM Data that will be used to generate ROM
    if verbosity >= 1:
        print("Getting Simulation Data")
    dataModelCoeff = np.empty((len(fomParamSamples),nT,model.nCollocation*model.nElements*2*neq))
    for i in range(len(fomParamSamples)):
        perturbedModel = model.copy(params=fomParamSamples[i])
        dydtSens =lambda y,t: perturbedModel.dydtSens(y,t,paramSelect=paramSelect)
        dataModelCoeff[i]= model.solve_ivp(lambda t,y: dydtSens(y,t),initialCondition)
        if verbosity >=3:
            print("dataModelCoeff shape: ", dataModelCoeff.shape)

    #Step 2: Get FOM Data that will be used to compare to ROM
    refModelCoeff = np.empty((len(romParamSamples),nT,model.nCollocation*model.nElements*2*neq))
    for i in range(len(romParamSamples)):
        perturbedModel = model.copy(params=romParamSamples[i])
        dydtSens =lambda y,t: perturbedModel.dydtSens(y,t,paramSelect=paramSelect)
        refModelCoeff[i]= model.solve_ivp(lambda t,y: dydtSens(y,t),initialCondition)
        if verbosity >=3:
            print("refModelCoeff shape: ", refModelCoeff.shape)
    #================================== Run POD-ROM ==================================================================================
    
    for iquad in range(len(quadRule)):
        for isens in range(len(romSensitivityApproach)):
            for iInit in range(len(sensInit)):
                if verbosity >= 1:
                    print(f"Using {quadRule[iquad]} quadrature rule, {romSensitivityApproach[isens]} sensitivity approach, {sensInit[iInit]} sensitivity initialization")
                for imean in range(len(mean_reduction)):
                    truncationError=np.empty((len(modeRetention),))
                    error = []
                    qoiResults = []
                    x,W = model.getQuadWeights(nPoints,quadRule[iquad])
                    uFomData=np.empty((len(fomParamSamples),neq,nT,nPoints))
                    vFomData=np.empty((len(fomParamSamples),neq,nT,nPoints))
                    uRefData=np.empty((len(romParamSamples),neq,nT,nPoints))
                    vRefData=np.empty((len(romParamSamples),neq,nT,nPoints))
                    for i in range(neq):
                        fomStart = i*(2*model.nCollocation*model.nElements)
                        for j in range(len(fomParamSamples)):
                            uFomData[j,i], vFomData[j,i] = model.eval(x,dataModelCoeff[j,:,fomStart:fomStart+2*model.nCollocation*model.nElements],output="seperated")
                        for j in range(len(romParamSamples)):
                            uRefData[j,i], vRefData[j,i] = model.eval(x,refModelCoeff[j,:,fomStart:fomStart+2*model.nCollocation*model.nElements],output="seperated")
                    for iret in range(len(modeRetention)):
                        #============================= Construct POD 
                        if usePodRom:
                            if equationSet != "tankOnly":
                                podSaveFolder = fomSaveFolder + "_" + romSensitivityApproach[isens] + "_" + sensInit[iInit]
                                if romSensitivityApproach[isens] == "finite":
                                    podSaveFolder += "_d"+str(finiteDelta)
                                elif romSensitivityApproach[isens] == "complex":
                                    podSaveFolder += "_d"+str(complexDelta)
                            else:
                                podSaveFolder = fomSaveFolder
                            if penaltyStrength != 0:
                                podSaveFolder += "/"+mean_reduction[imean] + "_"+quadRule[iquad]+"_n"+str(nPoints)+"_p"+str(penaltyStrength)
                            else:
                                podSaveFolder += "/"+mean_reduction[imean] + "_"+quadRule[iquad]+"_n"+str(nPoints)
                            
                            if not os.path.exists(podSaveFolder) and plotFullSpectra:
                                os.makedirs(podSaveFolder)

                            #Plot full singular value distribution
                            if iret ==0 and plotFullSpectra:
                                romData, null =model.constructPodRom(dataModelCoeff[:,:,:2*nCollocation*nElements],x,W,min(nT,nPoints),mean=mean_reduction[imean],useEnergyThreshold=False,quadPrecision=True)
                                fig, axes = plt.subplots(1,1, figsize=(5,4))
                                axes.loglog(romData.uSingularValues,"-b",lw=5,ms=8)
                                axes.loglog(romData.vSingularValues,"--m",lw=5,ms=8)
                                axes.legend(["u","v"])
                                axes.set_xlabel("Mode")
                                axes.set_ylabel("Singular Value")
                                plt.tight_layout()
                                plt.savefig(podSaveFolder + "../singularValues.pdf", format="pdf")
                                plt.savefig(podSaveFolder + "../singularValues.png", format="png")
                            #------------------------------- Compute POD
                            romData, truncationError[iret]=model.constructPodRom(dataModelCoeff[:,:,:2*nCollocation*nElements],x,W,modeRetention[iret],mean=mean_reduction[imean],useEnergyThreshold=useEnergyThreshold,adjustModePairs=False)
                    
                        error.append(np.empty((neq,len(romParamSamples),len(error_norm))))
                        qoiResults.append(np.empty((len(romParamSamples),len(qois)))) #Goal: Implement Computation of QoI sensitivity
                        if verbosity >=1:
                            print("             Running POD-ROM for mode retention ", modeRetention[iret], " and mean reduction ", mean_reduction[imean])

                        #Intialize results storage
                        if usePodRom:
                            uResults = np.empty((len(romParamSamples),neq,nT,3,nPoints))
                            vResults = np.empty((len(romParamSamples),neq,nT,3,nPoints))
                        else:
                            uResults = np.empty((len(romParamSamples),neq,nT,1,nPoints))
                            vResults = np.empty((len(romParamSamples),neq,nT,1,nPoints))
                        #Fill initial FOM data
                        uResults[:,:,:,0,:] = uRefData
                        vResults[:,:,:,0,:] = vRefData

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
                                romSaveFolder = controlSaveFolder + "e"+str(modeRetention[iret])+"/"
                            else :
                                romSaveFolder = controlSaveFolder + "m"+str(modeRetention[iret])+"/"

                            #Create folders if they don't exist
                            if not os.path.exists(controlSaveFolder) and plotConvergence:
                                os.makedirs(controlSaveFolder)
                            if not os.path.exists(romSaveFolder) and (plotTimeSeries or plotModes or plotError or plotRomCoeff or plotSingularValues or plotRomInterpolation):
                                os.makedirs(romSaveFolder)

                            #Update romData with control approach
                            if nDeimPoints != "max":
                                romData = model.computeDEIMProjection(romData, nDeimPoints)
                            if nonLinReduction >1.0:
                                romData = model.computeNonLinReduction(romData, nonLinReduction, proportionality = "min singular value")
                                                    #Compute time modes for sensitivity equations
                            
                            print(romData.uNonlinDim,", ", romData.vNonlinDim)

                        #======================================== Run POD-ROM ===============================================================================
                        for iParamSample in range(len(romParamSamples)):
                            if usePodRom:
                                if verbosity >=2:
                                    print("                 Running POD-ROM for parameter sample ", iParamSample+1, " of ", len(romParamSamples))
                                model = model.copy(params=romParamSamples[iParamSample])
                                dydtPodRom = lambda y,t: model.dydtPodRom(y,t,romData,paramSelect = [])
                                #------------------------------ Run ROM
                                #Get Initial Modal Values
                                # NOTE: Need to check how indexing is going to be coming out of the POD development. They will all have the same initial condition but I need to decide if it's a 2D or 3D array
                                romInit=np.empty((romData.uNmodes+romData.vNmodes))
                                romInit[:romData.uNmodes]\
                                    =romData.uTimeModes[0,:romData.uNmodes]
                                romInit[romData.uNmodes:romData.uNmodes+romData.vNmodes]\
                                    =romData.vTimeModes[0,:romData.vNmodes]
                                #Compute Base Rom Value
                                romCoeff = np.empty((nT,neq*(romData.uNmodes+romData.vNmodes)))
                                romCoeff[:,:romData.uNmodes+romData.vNmodes] = model.solve_ivp(lambda t,y: dydtPodRom(y,t),romInit)
                                
                                #------------------------------- Compute Sensitivity
                                #NOTE: Sensitivity not yet implemented for multiple parameter samples
                                if equationSet!="tankOnly":
                                    romCoeff = computeSensitivity(romCoeff,model,romData,paramSelect,romSensitivityApproach[isens],sensInit[iInit],finiteDelta=finiteDelta, complexDelta=complexDelta, verbosity = verbosity)

                                #----------------------------- Map Results Back into Spatial Space
                                for i in range(0, neq):
                                    # Compute ROM Solution
                                    romModeStart = i*(romData.uNmodes+romData.vNmodes)
                                    #ROM Result
                                    if i==0:
                                        uResults[iParamSample,i,:,1,:] = (romData.uModes @ romCoeff[:,romModeStart:romModeStart+romData.uNmodes].transpose()).transpose() + romData.uMean
                                        vResults[iParamSample,i,:,1,:] = (romData.vModes @ romCoeff[:,romModeStart+romData.uNmodes:romModeStart+romData.uNmodes+romData.vNmodes].transpose()).transpose() + romData.vMean
                                    else:
                                        #NOTE: We assume 0 mean decomposition for sensitivity equations. This is mathematically reasonable in the perturbed parameters case where the inital sensitivity is 0 anyways. 
                                        #   Additionally, we don't have a way of computing a more optimal mean decomposition in the imagined circumstance where FOM sensitivities weren't solved
                                        uResults[iParamSample,i,:,1,:] = (romData.uModes @ romCoeff[:,romModeStart:romModeStart+romData.uNmodes].transpose()).transpose()
                                        vResults[iParamSample,i,:,1,:] = (romData.vModes @ romCoeff[:,romModeStart+romData.uNmodes:romModeStart+romData.uNmodes+romData.vNmodes].transpose()).transpose()
                                    #POD Result
                                    #NOTE: The idea of the POD value at each of the romSample points doesn't make sense since the these data points weren't included in the POD decomposition. Maybe seperate the POD and ROM results arrays?
                                    #This line will currently fail because uTimeModes is much larger in dimension now so only calling it if we're in a single rom parameter case
                                    if len(romParamSamples)==1:
                                        if i==0:
                                            uResults[iParamSample,i,:,2,:] = ((romData.uModes @ romData.uTimeModes[:romCoeff.shape[0],:].transpose())+romData.uMean.reshape((romData.uMean.size,1))).transpose()
                                            vResults[iParamSample,i,:,2,:] = (romData.vModes @ romData.vTimeModes[:romCoeff.shape[0],:].transpose() +romData.vMean.reshape((romData.vMean.size,1))).transpose()
                                        else:
                                            #We can generalize POD to the variationin the POD space projected back to the FOM space, regardless of whether sensitivities were in initial POD decomp
                                            #Note: Confirmed that, if POD modes are computed using sensitivity snapshots, then this is equivalent to those modes if no mean decomp is used
                                            #UNVERIFIED: That that property holds numerically with this implementation and whether it holds if using a mean decomp
                                            if mean_reduction[imean]!="zero":
                                                print("WARNING: Correctness of approach unconfirmed for non-zero mean reduction")
                                            
                                            uResults[iParamSample,i,:,2,:] = (romData.uModes @ uFullTimeModes[:,i*romData.uNmodes:(i+1)*romData.uNmodes].transpose()+romData.uMean.reshape((romData.uMean.size,1))).transpose()
                                            vResults[iParamSample,i,:,2,:] = (romData.vModes @ vFullTimeModes[:,i*romData.vNmodes:(i+1)*romData.vNmodes].transpose()+romData.vMean.reshape((romData.vMean.size,1))).transpose()
                                #================================================== Compute Error =================================================================
                                for ieq in range(neq):
                                    for k in range(len(error_norm)):
                                        #Error for ith param sample, only domain equations, all times, all points, between FOM and ROM
                                        error[iret][ieq,iParamSample,k] = model.computeRomError(uResults[iParamSample,ieq,:,0,:].transpose(),vResults[iParamSample,ieq,:,0,:].transpose(),uResults[iParamSample,ieq,:,1,:].transpose(),vResults[iParamSample,ieq,:,1,:].transpose(),romData.W,tEval,norm=error_norm[k])
                                        if verbosity >= 2:
                                            print(                  "ROM Error in norm "+error_norm[k]+" for ieq " + str(ieq) + ": ", error[iret][ieq,iParamSample,k])
                                #================================================== Compute QOIs =================================================================
                                for k in range(len(qois)):
                                    #Error for ith param sample, only domain equations, all times, all points, between FOM and ROM
                                    qoiResults[iret][iParamSample,k] = model.computeQOIs(uResults[iParamSample,0,:,1,:].transpose(),vResults[iParamSample,0,:,1,:].transpose(),romData.W,tEval,qoi=qois[k])
                                    if verbosity >= 2:
                                        print(                  qois[k]+": ", qoiResults[iret][iParamSample,k])
                                    #INCOMPLETE: Figure out what want to compute for sensitivity error.

                        #=========================================== Make Plots ===================================================================
                        if usePodRom:
                            if len(romParamSamples)==1:
                                legends = ["FOM","ROM","POD"] 
                            else:
                                legends = ["FOM","ROM"]
                        else:
                            legends = ["FOM"]

                        #------------------------------------------- Make Interpolation Plots ----------------------------------------------------
                        if plotRomInterpolation and len(romParamSamples)>1:
                            rom_values = np.array([p[param] for p in romParamSamples])
                            fig, axes = plt.subplots(1,1, figsize=(4,3))
                            #Base ROM error
                            #L2 Error
                            axes.semilogy(rom_values, error[iret][0,:,0],"-bs",lw=3,ms=8)
                            #Linf Error
                            axes.semilogy(rom_values, error[iret][0,:,1],"--m*",lw=3,ms=8)
                            axes.set_xlabel(param)
                            axes.set_ylabel("Error")
                            axes.legend(error_norm)
                            plt.savefig(romSaveFolder + "OATaccuracy_"+param + "_a" + str(paramBounding) + "nSamp" + str(nRomSamples) + ".pdf", format="pdf")
                            plt.savefig(romSaveFolder + "OATaccuracy_"+param + "_a" + str(paramBounding) + "nSamp" + str(nRomSamples) + ".png", format="png")

                            #Sensitivity Error
                            for i in range(1, neq): 
                                fig, axes = plt.subplots(1,1, figsize=(4,3))
                                #L2 Error
                                axes.semilogy(rom_values, error[iret][i, :,0],"-bs",lw=3,ms=8)
                                #Linf Error
                                axes.semilogy(rom_values, error[iret][i, :,1],"--m*",lw=3,ms=8)
                                axes.set_xlabel(param)
                                axes.set_ylabel("Error")
                                axes.legend(error_norm)
                                plt.savefig(romSaveFolder + "OATsensAccuracy_"+param +"_" + str(romSensitivityApproach[isens]) + "_a" + str(paramBounding) + "nSamp" + str(nRomSamples) + ".pdf", format="pdf")
                                plt.savefig(romSaveFolder + "OATsensAccuracy_"+param +"_" + str(romSensitivityApproach[isens]) + "_a" + str(paramBounding) + "nSamp" + str(nRomSamples) + ".png", format="png")

                            fig, axes = plt.subplots(1,1, figsize=(4,3))
                            #L2 Error
                            axes.semilogy(rom_values, qoiResults[iret][ :,0],"-bs",lw=3,ms=8)
                            #Linf Error
                            axes.semilogy(rom_values, qoiResults[iret][ :,1],"--m*",lw=3,ms=8)
                            #L2 Error
                            axes.semilogy(rom_values, qoiResults[iret][ :,2],"-.g^",lw=3,ms=8)
                            #Linf Error
                            axes.semilogy(rom_values, qoiResults[iret][ :,3],"-ro",lw=3,ms=8)
                            axes.set_xlabel(param)
                            axes.legend(qois)
                            plt.savefig(romSaveFolder + "OATqois_"+param + "_a" + str(paramBounding) + "nSamp" + str(nRomSamples) + ".pdf", format="pdf")
                            plt.savefig(romSaveFolder + "OATqois_"+param + "_a" + str(paramBounding) + "nSamp" + str(nRomSamples) + ".png", format="png")
                        #------------------------------------------- Make Movies ----------------------
                        #Concatenate results for easier mangament in plotting 
                        combinedResults=np.empty((uResults.shape[0],2*neq)+uResults.shape[2:])
                        for i in range(1, neq):
                            combinedResults[:,2*i,:,:,:]=uResults[:,i,:,:,:]
                            combinedResults[:,2*i+1,:,:,:]=vResults[:,i,:,:,:]
                        if usePodRom and makeMovies:   
                            subplotMovie([u for u in uResults[0]], x, romSaveFolder + "u.mov", fps=15, xLabels="x", yLabels=uLabels, legends=legends, legendLoc="upper left", subplotSize=(2.5, 2))
                            subplotMovie([v for v in vResults[0]], x, romSaveFolder + "v.mov", fps=15, xLabels="x", yLabels=vLabels, legends=legends, legendLoc="upper left", subplotSize=(2.5, 2))
                            subplotMovie([y for y in combinedResults[0]], x, romSaveFolder + "combined.mov", fps=15, xLabels="x",  yLabels=combinedLabels, legends=legends, legendLoc="upper left", subplotSize=(2.5, 2))
                            if plotError:
                                subplotMovie([u[:,1:3,:]-u[:,[0],:] for u in uResults], x, romSaveFolder + "uError.mov", fps=15, xLabels="x", yLabels=uLabels, legends=legends[1:3], legendLoc="upper left", subplotSize=(2.5, 2),lineTypeStart=1,yRanges="auto")
                                subplotMovie([v[:,1:3,:]-v[:,[0],:] for v in vResults], x, romSaveFolder + "vError.mov", fps=15, xLabels="x", yLabels=vLabels, legends=legends[1:3], legendLoc="upper left", subplotSize=(2.5, 2),lineTypeStart=1,yRanges="auto")   
                        elif makeMovies:   
                            subplotMovie([u for u in uResults[0]], x, fomSaveFolder + "u.mov", fps=15, xLabels="x", yLabels=uLabels, legends=legends, legendLoc="upper left", subplotSize=(2.5, 2))
                            subplotMovie([v for v in vResults[0]], x, fomSaveFolder + "v.mov", fps=15, xLabels="x", yLabels=vLabels, legends=legends, legendLoc="upper left", subplotSize=(2.5, 2))
                            subplotMovie([y for y in combinedResults[0]], x, fomSaveFolder + "combined.mov", fps=15, xLabels="x",  yLabels=combinedLabels, legends=legends, legendLoc="upper left", subplotSize=(2.5, 2))

                        #------------------------------------------ Make example plots --------------------------------------------------------
                        if plotTimeSeries:
                            tplot = np.linspace(0,tEval.size-1,4,dtype=int)
                            title = ["t=" + str(round(1000*tEval[it])/1000) for it in tplot]

                            fig,axs = subplotTimeSeries([u[tplot,:,:] for u in uResults[0]], x, xLabels="x", yLabels=uLabels, title = title,legends=legends, subplotSize=(2.65, 2))
                            if usePodRom:
                                plt.savefig(romSaveFolder + "uTimeSeries.pdf", format="pdf")
                                plt.savefig(romSaveFolder + "uTimeSeries.png", format="png")
                            else:
                                plt.savefig(fomSaveFolder + "uTimeSeries.pdf", format="pdf")
                                plt.savefig(fomSaveFolder + "uTimeSeries.png", format="png")
                            fig,axs = subplotTimeSeries([v[tplot,:,:] for v in vResults[0]], x, xLabels="x", yLabels=vLabels, title = title,legends=legends, subplotSize=(2.65, 2))
                            if usePodRom:
                                plt.savefig(romSaveFolder + "vTimeSeries.pdf", format="pdf")
                                plt.savefig(romSaveFolder + "vTimeSeries.png", format="png")
                            else:
                                plt.savefig(fomSaveFolder + "vTimeSeries.pdf", format="pdf")
                                plt.savefig(fomSaveFolder + "vTimeSeries.png", format="png")
                            fig,axs = subplotTimeSeries([y[tplot,:,:] for y in combinedResults[0]], x, xLabels="x", yLabels=combinedLabels, title = title,legends=legends, subplotSize=(2.65, 2))
                            if usePodRom:
                                plt.savefig(romSaveFolder + "combinedTimeSeries.pdf", format="pdf")
                                plt.savefig(romSaveFolder + "combinedTimeSeries.png", format="png")
                            else:
                                plt.savefig(fomSaveFolder + "combinedTimeSeries.pdf", format="pdf")
                                plt.savefig(fomSaveFolder + "combinedTimeSeries.png", format="png")
                            if usePodRom and plotError:
                                fig,axs = subplotTimeSeries([u[tplot,1:3,:]-u[tplot,0:1,:] for u in uResults[0]], x, xLabels="x", yLabels=uLabels, title = title,legends=legends[1:3], subplotSize=(2.65, 2),lineTypeStart=1)
                                plt.savefig(romSaveFolder + "uErrorTimeSeries.pdf", format="pdf")
                                plt.savefig(romSaveFolder + "uErrorTimeSeries.png", format="png")
                                fig,axs = subplotTimeSeries([v[tplot,1:3,:]-v[tplot,0:1,:] for v in vResults[0]], x, xLabels="x", yLabels=vLabels, title = title,legends=legends[1:3], subplotSize=(2.65, 2),lineTypeStart=1)
                                plt.savefig(romSaveFolder + "vErrorTimeSeries.pdf", format="pdf")
                                plt.savefig(romSaveFolder + "vErrorTimeSeries.png", format="png")
                            if plotRomCoeff:
                                coeffLabels = ["Coeff " + str(i+1) for i in range(romData.uNmodes)]
                                uCoeffData=[np.array([rom,pod]) for pod,rom in zip(romData.uTimeModes[:,:romData.uNmodes].transpose(),\
                                                                                romCoeff[:,:romData.uNmodes].transpose())]
                                
                                fig,axs = subplot(uCoeffData, tEval, xLabels="t", yLabels=coeffLabels, legends=legends[1:3], subplotSize=(2.65, 2),lineTypeStart=1)
                                plt.savefig(romSaveFolder + "uRomCoeff.pdf", format="pdf")
                                plt.savefig(romSaveFolder + "uRomCoeff.png", format="png")

                                coeffLabels = ["Coeff " + str(i+1) for i in range(romData.vNmodes)]
                                vCoeffData=[np.array([rom,pod]) for pod,rom in zip(romData.vTimeModes[:,:romData.vNmodes].transpose(),\
                                                                                romCoeff[:,romData.uNmodes:romData.uNmodes+romData.vNmodes].transpose())]
                                fig,axs = subplot(vCoeffData, tEval, xLabels="t", yLabels=coeffLabels,legends=legends[1:3], subplotSize=(2.65, 2),lineTypeStart=1)
                                plt.savefig(romSaveFolder + "vRomCoeff.pdf", format="pdf")
                                plt.savefig(romSaveFolder + "vRomCoeff.png", format="png")
                            if plotModes:
                                #Compute linearization of nonlinear term
                                nonLinearTerm = lambda u, v: baseParams["Da"]*romData.vModesWeighted.transpose()\
                                            @((1-(romData.uModes@u+romData.uMean))*np.exp(baseParams["gamma"]*baseParams["beta"]\
                                            *(romData.vModes@v+romData.vMean)/(1+baseParams["beta"]*(romData.vModes@v+romData.vMean))))
                                nonLinearLinearized = [None]*tplot.size
                                for it in range(tplot.size):
                                    nonLinearLinearized[it] =np.empty((romData.vNmodes,romData.vNmodes))
                                    baseU=romCoeff[tplot[it],:romData.uNmodes]
                                    baseV=romCoeff[tplot[it],romData.uNmodes:romData.uNmodes+romData.vNmodes]
                                    for i in range(romData.vNmodes):
                                        adjustedV=baseV.copy()
                                        adjustedV[i]+=1e-6
                                        if verbosity >= 3:
                                            print("Difference between adjusted and standard v basis:", adjustedV-baseV)
                                            print("Nonlinear evaluations with standard basis:", nonLinearTerm(baseU,baseV))
                                            print("Difference between nonlinear evaluations:", (nonLinearTerm(baseU,adjustedV)-nonLinearTerm(baseU,baseV))/1e-6)
                                        nonLinearLinearized[it][i,:]=(nonLinearTerm(baseU,adjustedV)-nonLinearTerm(baseU,baseV))/1e-6
                                
                                fig, axes = plotRomMatrices(nonLinearLinearized,\
                                                            xLabels=r"$v_j$",yLabels=[r"$\frac{\partial \mathcal{N}(v_i)}{\partial v_j}$"," "," ", " "],\
                                                            title=["t=" + str(round(1000*tEval[it])/1000) for it in tplot],\
                                                            cmap="coolwarm",
                                                            fontsize=18)
                                plt.savefig(romSaveFolder + "vRomMatrices_linearization.pdf", format="pdf", bbox_inches='tight', pad_inches=0.02)
                                plt.savefig(romSaveFolder + "vRomMatrices_linearization.png", format="png", bbox_inches='tight', pad_inches=0.02)

                        #------------------------------------------ Plot POD Modes --------------------------------------------------------
                        if plotModes and usePodRom:
                            subplot([mode for mode in romData.uModes.transpose()], x, xLabels="x", yLabels=["Mode " + str(i+1) for i in range(romData.uNmodes)])
                            plt.savefig(romSaveFolder + "uModes.pdf", format="pdf")
                            plt.savefig(romSaveFolder + "uModes.png", format="png")
                            subplot([mode for mode in romData.vModes.transpose()], x, xLabels="x", yLabels=["Mode " + str(i+1) for i in range(romData.vNmodes)])
                            plt.savefig(romSaveFolder + "vModes.pdf", format="pdf")
                            plt.savefig(romSaveFolder + "vModes.png", format="png")


                            fig, axes = plotRomMatrices(romData.vRomFirstOrderMat,\
                                                        xLabels=r"$\phi'_{v,j}$",yLabels=r"$\phi_{v,i}$",\
                                                        title=r"1st Order v ROM matrix: $\langle\phi_{v,i},\phi'_{v,j}\rangle_{L^2}$",\
                                                        cmap="coolwarm")
                            plt.savefig(romSaveFolder + "vRomMatrices_1stOrder.pdf", format="pdf")
                            plt.savefig(romSaveFolder + "vRomMatrices_1stOrder.png", format="png")
                            fig, axes = plotRomMatrices(romData.vRomSecondOrderMat,\
                                                        xLabels=r"$\phi''_{v,j}$",yLabels=r"$\phi_{v,i}$",\
                                                        title=r"2nd Order v ROM matrix: $\langle\phi_{v,i},\phi''_{v,j}\rangle_{L^2}$",\
                                                        cmap="coolwarm")
                            plt.savefig(romSaveFolder + "vRomMatrices_2ndOrder.pdf", format="pdf")
                            plt.savefig(romSaveFolder + "vRomMatrices_2ndOrder.png", format="png")
                            fig, axes = plotRomMatrices([romData.vRomFirstOrderMat,romData.vRomSecondOrderMat],\
                                                        xLabels=[r"$\phi'_{v,j}$",r"$\phi''_{v,j}$"],yLabels=r"$\phi_{v,i}$",\
                                                        title=[r"1st Order v ROM matrix: $\langle\phi_{v,i},\phi'_{v,j}\rangle_{L^2}$",\
                                                            r"2nd Order v ROM matrix: $\langle\phi_{v,i},\phi''_{v,j}\rangle_{L^2}$"],\
                                                        cmap="coolwarm")
                            plt.savefig(romSaveFolder + "vRomMatrices.pdf", format="pdf")
                            plt.savefig(romSaveFolder + "vRomMatrices.png", format="png")

                            fig, axes = plotRomMatrices(romData.uRomFirstOrderMat,\
                                                        xLabels=r"$\phi'_{u,j}$",yLabels=r"$\phi_{u,i}$",\
                                                        title=r"1st Order u ROM matrix: $\langle\phi_{u,i},\phi'_{u,j}\rangle_{L^2}$",\
                                                        cmap="coolwarm")
                            plt.savefig(romSaveFolder + "uRomMatrices_1stOrder.pdf", format="pdf")
                            plt.savefig(romSaveFolder + "uRomMatrices_1stOrder.png", format="png")
                            fig, axes = plotRomMatrices(romData.uRomSecondOrderMat,\
                                                        xLabels=r"$\phi''_{u,j}$",yLabels=r"$\phi_{u,i}$",\
                                                        title=r"2nd Order u ROM matrix: $\langle\phi_{u,i},\phi''_{u,j}\rangle_{L^2}$",\
                                                        cmap="coolwarm")
                            plt.savefig(romSaveFolder + "uRomMatrices_2ndOrder.pdf", format="pdf")
                            plt.savefig(romSaveFolder + "uRomMatrices_2ndOrder.png", format="png")
                            fig, axes = plotRomMatrices([romData.uRomFirstOrderMat,romData.uRomSecondOrderMat],\
                                                        xLabels=[r"$\phi'_{u,j}$",r"$\phi''_{u,j}$"],yLabels=r"$\phi_{u,i}$",\
                                                        title=[r"1st Order u ROM matrix: $\langle\phi_{u,i},\phi'_{u,j}\rangle_{L^2}$",\
                                                            r"2nd Order u ROM matrix: $\langle\phi_{u,i},\phi''_{u,j}\rangle_{L^2}$"],\
                                                        cmap="coolwarm")
                            plt.savefig(romSaveFolder + "uRomMatrices.pdf", format="pdf")
                            plt.savefig(romSaveFolder + "uRomMatrices.png", format="png")
                            

                        #------------------------------------------- Plot Singular Values --------------------------------------------------------
                        if plotSingularValues:
                            #Compute culmulative truncation
                            nSV = max(romData.uSingularValues.size, romData.vSingularValues.size)
                            totalTruncation = 1 - np.cumsum(np.pad(romData.uSingularValues,(0, nSV-romData.uSingularValues.size))
                                                            +np.pad(romData.vSingularValues,(0, nSV-romData.vSingularValues.size))) \
                                                    / np.sum(romData.uFullSpectra+romData.vFullSpectra)
                            # uCutoffIndices = np.array([romData.uSingularValues.size-np.sum(uPropInformation<=.1),
                            #                            romData.uSingularValues.size-np.sum(uPropInformation<=.01),
                            #                            romData.uSingularValues.size-np.sum(uPropInformation<=.001),
                            #                            romData.uSingularValues.size-np.sum(uPropInformation<=.0001)])
                            # vCutoffIndices = np.array([romData.vSingularValues.size-np.sum(vPropInformation<=.1),
                            #                            romData.vSingularValues.size-np.sum(vPropInformation<=.01),
                            #                            romData.vSingularValues.size-np.sum(vPropInformation<=.001),
                            #                            romData.vSingularValues.size-np.sum(vPropInformation<=.0001)])
                            cutoffIndices = np.array([nSV - np.sum(totalTruncation <= .1),
                                                        nSV - np.sum(totalTruncation <= .01),
                                                        nSV - np.sum(totalTruncation <= .001),
                                                        nSV - np.sum(totalTruncation <= .0001)])
                            cutoffIndices = cutoffIndices[(cutoffIndices<romData.uSingularValues.size) & (cutoffIndices<romData.vSingularValues.size)]
                            uCutoffSv = romData.uSingularValues[cutoffIndices]
                            vCutoffSv = romData.vSingularValues[cutoffIndices]
                            padding = 1.3
                            width = .95
                            cutoffSV_padded = np.array([np.maximum(uCutoffSv,vCutoffSv)*padding, np.minimum(uCutoffSv,vCutoffSv)/padding]).T
                            fig, axes = plt.subplots(1,1, figsize=(5,4))
                            axes.semilogy(np.arange(1,romData.uSingularValues.size+1),romData.uSingularValues,"bs",lw=5,ms=5,)
                            axes.semilogy(np.arange(1,romData.vSingularValues.size+1),romData.vSingularValues,"mo",lw=5,ms=5,)
                            labels = ["90%", "99%", "99.9%", "99.99%"]
                            for i in range(cutoffIndices.size):
                                rect = Rectangle((cutoffIndices[i]+1 - width/2, cutoffSV_padded[i,1]), width, cutoffSV_padded[i,0]-cutoffSV_padded[i,1], 
                                                facecolor='none', edgecolor='k', linewidth=2, zorder=3)
                                plt.gca().add_patch(rect)
                                # Place label just above the TOP edge of the rectangle (robust for log-scale)
                                y_top = cutoffSV_padded[i, 0]
                                x_right = cutoffIndices[i] + 1 + width/2
                                # Use a small screen-space offset so it always appears above the line regardless of scale
                                axes.annotate(labels[i], xy=(x_right, y_top), xytext=(0, 4), textcoords='offset points',
                                                ha='center', va='bottom', fontsize=10, color='k', zorder=5)

                            # Create legend for first and third lines (indices 0 and 2)
                            plt.legend(['u', 'v'])

                            axes.set_xlabel("Mode")
                            axes.set_ylabel("Singular Value")
                            plt.tight_layout()
                            plt.savefig(romSaveFolder + "singularValues.pdf", format="pdf")
                            plt.savefig(romSaveFolder + "singularValues.png", format="png")

                            fig, axes = plt.subplots(1,1, figsize=(5,4))
                            uPropInformation = 1 - np.cumsum(romData.uSingularValues)/np.sum(romData.uFullSpectra)
                            vPropInformation = 1 - np.cumsum(romData.vSingularValues)/np.sum(romData.vFullSpectra)
                            axes.semilogy(np.arange(1,romData.uSingularValues.size+1),uPropInformation[:romData.uSingularValues.size],"bs",lw=5,ms=8)
                            axes.semilogy(np.arange(1,romData.vSingularValues.size+1),vPropInformation[:romData.vSingularValues.size],"mo",lw=5,ms=8)
                            axes.legend(["u","v"])
                            axes.set_xlabel("Mode")
                            axes.set_ylabel("Singular Value")
                            plt.tight_layout()
                            plt.savefig(romSaveFolder + "propInformation.pdf", format="pdf")
                            plt.savefig(romSaveFolder + "propInformation.png", format="png")
                        if not showPlots:
                            plt.close()
                #=========================================== Plot Convergence ===================================================================
                    
                    if usePodRom and plotConvergence and len(error)>1 and not adaptiveControlCutoff:
                        #Don't plot error convergence for sensitivities or multiple parameter values
                        errorPlot = np.array([errorRet[0,0,:] for errorRet in error]).T.tolist()
                        errorPlot = [np.array(error) for error in errorPlot]
                        #Potential Error: We've refactored plotErrorConvergnece for lists of errors at each retention but this indexing doesn't achieve that
                        if len(mean_reduction)>1:
                            legends = [mean_method +", "+ norm for mean_method in mean_reduction for norm in error_norm]
                        else:
                            legends =error_norm
                        fig,axs = plotErrorConvergence(errorPlot,truncationError,xLabel="Proportion Information Truncated in POD",yLabel="Relative ROM Error",legends=legends) 
                        plt.savefig(controlSaveFolder + "errorConvergence_s"+str(modeRetention[0])+"_e" + str(modeRetention[-1])+".pdf", format="pdf")
                        plt.savefig(controlSaveFolder + "errorConvergence_s"+str(modeRetention[0])+"_e" + str(modeRetention[-1])+".png", format="png")
                    
                                

    if showPlots:
        plt.show()


if __name__ == "__main__":
    main()