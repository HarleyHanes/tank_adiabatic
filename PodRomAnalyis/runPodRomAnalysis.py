import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath(os.path.join(current_script_dir,'..',))
sys.path.append(grandparent_dir)
print(sys.path)
import numpy as np
import scipy
from postProcessing.plot import subplotTimeSeries
from postProcessing.plot import subplot
from postProcessing.plot import subplotMovie
from postProcessing.plot import plotErrorConvergence
from tankModel.TankModel import TankModel
import matplotlib.pyplot as plt


#Set run details
#FOM parameters
paramSet = "BizonPeriodic" #BizonPeriodic, BizonLinear, BizonNonLinear, BizonAdvecDiffusion
equationSet = "nonBoundaryParams" #tankOnly, Le, vH, linearParams, linearBoundaryParams, allParams, nonBoundaryParams
coarse = False
nCollocation=2
nElements=64

#romParameters
usePodRom=True
romSensitivityApproach = ["complex","sensEq"] #none, finite, sensEq, DEPOD (unfinished), complex, only used if equationSet!=tankOnly
finiteDelta = 1e-6   #Only used if equationSet!=tankOnly and romSensitivityApproach=="finite"
complexDelta = 1e-10 #Only used if equationSet!=tankOnly and romSensitivityApproach=="complex"
useEnergyThreshold=True
if useEnergyThreshold==True:
    #energyRetention=[.8,.85,.885,.925,.95,.975,.99,.9925,.995,.9975,.999,.9999]
    #modeRetention=[.85,.99,.999]
    modeRetention=[.999]
else:
    #modeRetention=[1,2,3,4,5,6,7]
    #modeRetention = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] #Stable for periodic with no mean-decomp, gauss-legendre points
    modeRetention = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24] #All modes (stable for periodic with no mean-decomp)
    #modeRetention = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24] #Chaotic, no mean-decomp (note 12 is not used due to instability in the POD-ROM)
    #modeRetention = [3,4,5,7,9,10,11,13,14,15,16,17,18,19,20,21,22,23] #Chaotic, mean decomp (note many smaller, even modes are not used due to instability in the POD-ROM)
    #modeRetention = [14,15,16,17,18,19,20,21,22,23] #Chaotic, first decomp (note many smaller, even modes are not used due to instability in the POD-ROM)
nPoints=99
sensInit = ["pod"]
quadRule = "simpson"
mean_reduction = ["zero"]
error_norm = [r"$L_2$",r"$L_\infty$"]

#Display settings
plotTimeSeries=True
plotModes=True
plotConvergence=False
plotError=True
makeMovies=False



if paramSet == "BizonChaotic":
    baseParams={"PeM": 700, "PeT": 700, "f": .3, "Le": 1, "Da": .15, "beta": 1.8, "gamma": 10,"delta": 2, "vH":-.065}
    stabalizationTime=150
    if coarse:
        tstep=.2
    else:
        tstep=.02
    tmax=4.1
elif paramSet == "BizonPeriodic":
    baseParams={"PeM": 300, "PeT": 300, "f": .3, "Le": 1, "Da": .15, "beta": 1.4, "gamma": 10,"delta": 2, "vH":-.045}
    stabalizationTime=20
    if coarse:
        tstep=.2
    else:
        tstep=.02
    tmax=2.1
elif paramSet == "BizonNonLinear":
    baseParams={"PeM": 300, "PeT": 300, "f": .3, "Le": 1, "Da": .15, "beta": 1.4, "gamma": 10,"delta": 0, "vH":0}
    stabalizationTime=.1
    if coarse:
        tstep=.4
    else:
        tstep=.02
    tmax=2
elif paramSet == "BizonLinear":
    baseParams={"PeM": 300, "PeT": 300, "f": 0, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 2, "vH":-.045}
    stabalizationTime=.1
    if coarse:
        tstep=.2
    else:
        tstep=.02
    tmax=2
elif paramSet == "BizonAdvecDiffusion":
    baseParams={"PeM": 1, "PeT": 1, "f": 0, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 0, "vH":0}
    stabalizationTime=.1
    if corase:
        tstep=.1
    else:
        tstep=.01
    tmax=1.5
elif paramSet == "BizonAdvec":
    baseParams={"PeM": 1e16, "PeT": 1e16, "f": 0, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 0, "vH":0}
    stabalizationTime=.1
    if coarse:
        tstep=.1
    else:
        tstep=.01
    tmax=.9
else: 
    raise ValueError("Invalid paramSet entered: " + str(equationSet))
    
fomSaveFolder = "../../results/podRomAnalysis/"+paramSet +"_nCol" + str(nCollocation) + "_nElem"+str(nElements)
if coarse:
    fomSaveFolder += "_coarse"
fomSaveFolder+="/"+equationSet+"/"
        


#Simulation Settings
#Determine parameters to get sensitivity of
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
if not os.path.exists(fomSaveFolder):
        os.makedirs(fomSaveFolder)
#==================================== Setup system ===============================================================================
print("Setting up system")
model=TankModel(nCollocation=nCollocation,nElements=nElements,spacing="legendre",bounds=[0,1],params=baseParams)
dydtSens =lambda y,t: model.dydtSens(y,t,paramSelect=paramSelect)
if paramSet=="BizonChaotic":
    #For Chaotic, need to run two stabalizations, one to get to the pre-periodic stable case, and another to get to the chaotic case
    stabalizationParams = baseParams.copy()
    stabalizationParams["vH"]=-.045
    stabalizationModel=TankModel(nCollocation=nCollocation,nElements=nElements,spacing="legendre",bounds=[0,1],params=stabalizationParams)
    modelCoeff=np.ones((1,model.nCollocation*model.nElements*2*neq))*-.045
    #Run out till stabalizing in periodic domain
    stabalizationDydtSens =lambda y,t: stabalizationModel.dydtSens(y,t,paramSelect=paramSelect)
    odeOut= scipy.integrate.solve_ivp(lambda t,y: stabalizationDydtSens(y,t),(0,stabalizationTime),modelCoeff[-1,:], method='BDF',atol=1e-6,rtol=1e-6)
    modelCoeff = odeOut.y[:,[-1]].transpose()
elif paramSet=="BizonAdvec":
    collPoints=np.append(model.collocationPoints,model.collocationPoints,axis=0)
    modelCoeff=np.array([3/2*collPoints**2-collPoints**3])
    for i in range(neq-1):
        #modelCoeff = np.append(modelCoeff,[-collPoints**2+2*collPoints],axis=1)
        modelCoeff = np.append(modelCoeff,[0*model.collocationPoints],axis=1)
        modelCoeff = np.append(modelCoeff,[3/2*model.collocationPoints**3-model.collocationPoints**3],axis=1)
    print(modelCoeff.shape)
elif baseParams["vH"]==0:
    modelCoeff=np.ones((1,model.nCollocation*model.nElements*2*neq))*.5
else:
    modelCoeff=np.ones((1,model.nCollocation*model.nElements*2*neq))*baseParams["vH"]

#=================================== Run Stabalization ===========================================================================
print("Running Stabalization")
if not stabalizationTime==0:
    #Run out till stabalizing in periodic domain
    odeOut= scipy.integrate.solve_ivp(lambda t,y: dydtSens(y,t),(0,stabalizationTime),modelCoeff[-1,:], method='BDF',atol=1e-6,rtol=1e-6)
    modelCoeff = odeOut.y[:,[-1]].transpose()
#=================================== Get Simulation Data ================================================================
print("Getting Simulation Data")
t=0
while t<tmax:
    odeOut= scipy.integrate.solve_ivp(lambda t,y: dydtSens(y,t),(t,t+tstep),modelCoeff[-1], method='BDF',atol=1e-10,rtol=1e-10)
    change = np.max(np.abs(odeOut.y[:,-1]-modelCoeff[-1,:]))
    modelCoeff=np.append(modelCoeff,odeOut.y[:,[-1]].transpose(),axis=0)
    #print("(t,change): (", t,", ", change, ")")
    t+=tstep
    print(t)
#pre-allocation results storage
if usePodRom:
    uResults = np.empty((neq,modelCoeff.shape[0],3,nPoints))
    vResults = np.empty((neq,modelCoeff.shape[0],3,nPoints))
    combinedResults=np.empty((2*neq,modelCoeff.shape[0],3,nPoints))
else:
    uResults = np.empty((neq,modelCoeff.shape[0],1,nPoints))
    vResults = np.empty((neq,modelCoeff.shape[0],1,nPoints))
    combinedResults=np.empty((2*neq,modelCoeff.shape[0],1,nPoints))
#Change all POD-ROM parameters to lists if not  already
if type(modeRetention)==float or type(modeRetention)==int:
    modeRetention = [modeRetention]
if type(mean_reduction)==str:
    mean_reduction = [mean_reduction]
if type(romSensitivityApproach)==str:
    romSensitivityApproach = [romSensitivityApproach]
if type(quadRule)==str:
    quadRule = [quadRule]
if type(sensInit)==str:
    sensInit = [sensInit]



#================================== Run POD-ROM ==================================================================================
 
for iquad in range(len(quadRule)):
    x,W = model.getQuadWeights(nPoints,quadRule[iquad])
    #--------------------------------- Map results back to spatial points
    for i in range(0, neq):
        fomStart = i*(2*model.nCollocation*model.nElements)
        uFomResult, vFomResult = model.eval(x,modelCoeff[:,fomStart:fomStart+2*model.nCollocation*model.nElements],output="seperated")
        uResults[i,:,0,:]=uFomResult
        vResults[i,:,0,:]=vFomResult
        combinedResults[2*i,:,0,:]=uFomResult
        combinedResults[2*i+1,:,0,:]=vFomResult
    for isens in range(len(romSensitivityApproach)):
        for iInit in range(len(sensInit)):
            error = np.empty((len(modeRetention),len(mean_reduction),len(error_norm)))
            truncationError=np.empty((len(modeRetention),len(mean_reduction),len(error_norm)))
            for imean in range(len(mean_reduction)):
                for iret in range(len(modeRetention)):
                    if usePodRom:
                        print("Running POD-ROM for mode retention ", modeRetention[iret], " and mean reduction ", mean_reduction[imean])
                        #------------------------------- Make Folder to save data
                        if equationSet != "tankOnly":
                            romSaveFolder = fomSaveFolder +  romSensitivityApproach[isens] + "_" + sensInit[iInit]
                            if romSensitivityApproach[isens] == "finite":
                                romSaveFolder += "_d"+str(finiteDelta)
                            elif romSensitivityApproach[isens] == "complex":
                                romSaveFolder += "_d"+str(complexDelta)
                        else:
                            romSaveFolder = fomSaveFolder
                        romSaveFolder += "/"+mean_reduction[imean] + "_"+quadRule[iquad]+"_n"+str(nPoints)+"/"
                        if useEnergyThreshold:
                            romSaveFolder += "e"+str(modeRetention[iret])+ "/"
                        else :
                            romSaveFolder += "m"+str(modeRetention[iret])+ "/"

                        if not os.path.exists(romSaveFolder):
                            os.makedirs(romSaveFolder)

                        #------------------------------- Compute POD
                        # Get POD Decomposition
                        if romSensitivityApproach[isens] == "finite" or romSensitivityApproach[isens] == "complex" or romSensitivityApproach[isens] == "none":
                            #No adjustment to POD Needed for finite difference approach
                            romData, truncationError[iret,imean,:]=model.constructPodRom(modelCoeff[:,:2*nCollocation*nElements],x,W,modeRetention[iret],mean=mean_reduction[imean],useEnergyThreshold=useEnergyThreshold)
                        elif romSensitivityApproach[isens] == "DEPOD":
                            # INCOMPLETE: need to setup POD computation for mixed-solution and sensitivity snapshot matrices
                            raise ValueError("DEPOD not yet implemented")
                            romData, truncationError[iret,imean,:]=model.constructPodRom(modelCoeff,x,modeRetention[iret],mean=mean_reduction[imean],useEnergyThreshold=useEnergyThreshold)
                        elif romSensitivityApproach[isens] == "sensEq":
                            romData, truncationError[iret,imean,:]=model.constructPodRom(modelCoeff[:,:2*nCollocation*nElements],x,W,modeRetention[iret],mean=mean_reduction[imean],useEnergyThreshold=useEnergyThreshold)
                        else:
                            raise ValueError("Invalid romSensitivityApproach entered: " + str(romSensitivityApproach[isens]))
                        #Compute time modes for sensitivity equations
                        if romSensitivityApproach[isens] != "DEPOD":
                            uFullTimeModes = romData.uTimeModes.copy()
                            vFullTimeModes = romData.vTimeModes.copy()
                            for i in range(neq-1):
                                uFullTimeModes = np.append(uFullTimeModes,\
                                                        (romData.uModesWeighted.transpose() @ uResults[i+1,:,0,:].transpose()).transpose(),\
                                                        axis=1)
                                vFullTimeModes = np.append(vFullTimeModes,\
                                                        (romData.vModesWeighted.transpose() @ vResults[i+1,:,0,:].transpose()).transpose(),\
                                                        axis=1)
                            #Check Time modes match for first nModes
                            if not np.isclose(uFullTimeModes[:,:romData.uNmodes],romData.uTimeModes).all():
                                raise ValueError("u time modes do not match")
                            if not np.isclose(vFullTimeModes[:,:romData.vNmodes],romData.vTimeModes).all():
                                raise ValueError("v time modes do not match")

                        #------------------------------ Run POD-ROM
                        #Get Initial Modal Weights
                        if romSensitivityApproach[isens] == "finite" or romSensitivityApproach[isens] == "complex" or romSensitivityApproach[isens] == "none":
                            romCoeff=np.empty((1,romData.uNmodes+romData.vNmodes))
                            romCoeff[0,:romData.uNmodes]\
                                =romData.uTimeModes[0,:romData.uNmodes]
                            romCoeff[0,romData.uNmodes:romData.uNmodes+romData.vNmodes]\
                                =romData.vTimeModes[0,:romData.vNmodes]
                            dydtPodRom = lambda y,t: model.dydtPodRom(y,t,romData,paramSelect = [],penaltyStrength=0)
                        elif romSensitivityApproach[isens] == "DEPOD":
                            # INCOMPLETE: need to setup POD computation for mixed-solution and sensitivity snapshot matrices
                            raise ValueError("DEPOD not yet implemented")
                            #Compute Time modes for sensitivity equations
                            for i in range(1, neq):
                                start = i*(2*model.nCollocation*model.nElements)
                                timeModeIndex = modelCoeff.shape[0]*i
                                romCoeff[0,start:start+romData.uNmodes]\
                                    =romData.uTimeModes[timeModeIndex,:romData.uNmodes]
                                romCoeff[0,start+romData.uNmodes:start+romData.uNmodes+romData.vNmodes]\
                                    =romData.vTimeModes[timeModeIndex,:romData.vNmodes]
                            dydtPodRom = lambda y,t: model.dydtPodRom(y,t,romData,paramSelect=paramSelect,penaltyStrength=0)
                        elif romSensitivityApproach[isens] == "sensEq":
                            romCoeff=np.empty((1,neq*(romData.uNmodes+romData.vNmodes)))
                            for ieq in range(neq):
                                start = ieq*(romData.uNmodes+romData.vNmodes)
                                if ieq == 0 or sensInit[iInit]=="pod":
                                    romCoeff[0,start:start+romData.uNmodes]\
                                        =uFullTimeModes[0,ieq*romData.uNmodes:(ieq+1)*romData.uNmodes]
                                    romCoeff[0,start+romData.uNmodes:start+romData.uNmodes+romData.vNmodes]\
                                        =vFullTimeModes[0,ieq*romData.vNmodes:(ieq+1)*romData.vNmodes]
                                elif sensInit[iInit]=="zero":
                                    romCoeff[0,start:start+romData.uNmodes+romData.vNmodes] =np.zeros((1,romData.uNmodes+romData.vNmodes))
                                else:
                                    raise ValueError("Invalid sensInit entered: " + str(sensInit[iInit]))
                                dydtPodRom = lambda y,t: model.dydtPodRom(y,t,romData,paramSelect=paramSelect,penaltyStrength=0)
                        t=0
                        while t<tmax:
                            odeOut= scipy.integrate.solve_ivp(lambda t,y: dydtPodRom(y,t),(t,t+tstep),romCoeff[-1], method='BDF',atol=1e-10,rtol=1e-10)
                            change = np.max(np.abs(odeOut.y[:,-1]-romCoeff[-1,:]))
                            romCoeff=np.append(romCoeff,odeOut.y[:,[-1]].transpose(),axis=0)
                            t+=tstep

                        
                        if romSensitivityApproach[isens] == "finite":
                            for iparam in range(len(paramSelect)):
                                if sensInit[iInit]=="pod":
                                    #If finite-diff is exact at t=0 then u(0,x+delta)=u(0,x)+delta*u'(0,x)
                                    perturbedRomCoeff = romCoeff[[0],:romData.uNmodes+romData.vNmodes]
                                    perturbedRomCoeff[0,:romData.uNmodes] +=\
                                        finiteDelta*uFullTimeModes[0,(iparam+1)*romData.uNmodes:(iparam+2)*romData.uNmodes]
                                    perturbedRomCoeff[0,romData.uNmodes:romData.uNmodes+romData.vNmodes] += \
                                        finiteDelta*vFullTimeModes[0,(iparam+1)*romData.vNmodes:(iparam+2)*romData.vNmodes]
                                elif sensInit[iInit]=="zero":
                                    #If finite-diff is zero at t=0 then u(0,x+delta)=u(0,x)
                                    perturbedRomCoeff = romCoeff[[0],:romData.uNmodes+romData.vNmodes]
                                perturbedParams = baseParams.copy()
                                perturbedParams[paramSelect[iparam]]+= finiteDelta
                                perturbedModel=TankModel(nCollocation=nCollocation,nElements=nElements,spacing="legendre",bounds=[0,1],params=perturbedParams)
                                dydtPodRom = lambda y,t: perturbedModel.dydtPodRom(y,t,romData,paramSelect = [],penaltyStrength=0)
                                t=0
                                while t<tmax:
                                    odeOut= scipy.integrate.solve_ivp(lambda t,y: dydtPodRom(y,t),(t,t+tstep),perturbedRomCoeff[-1], method='BDF',atol=1e-10,rtol=1e-10)
                                    change = np.max(np.abs(odeOut.y[:,-1]-perturbedRomCoeff[-1,:]))
                                    perturbedRomCoeff=np.append(perturbedRomCoeff,odeOut.y[:,[-1]].transpose(),axis=0)
                                    t+=tstep
                                #Compute Sensitivity in POD space
                                romCoeff = np.append(romCoeff,(perturbedRomCoeff-romCoeff[:,:romData.uNmodes+romData.vNmodes])/finiteDelta,axis=1)
                        elif romSensitivityApproach[isens] == "complex":
                            for iparam in range(len(paramSelect)):
                                #Initialize sensitivity
                                if sensInit[iInit]=="pod":
                                    #If finite-diff is exact at t=0 then u(0,x+delta)=u(0,x)+delta*u'(0,x)
                                    perturbedRomCoeff = romCoeff[[0],:romData.uNmodes+romData.vNmodes].astype(complex)
                                    perturbedRomCoeff[0,:romData.uNmodes] += \
                                        1j*complexDelta*uFullTimeModes[0,(iparam+1)*romData.uNmodes:(iparam+2)*romData.uNmodes]
                                    perturbedRomCoeff[0,romData.uNmodes:romData.uNmodes+romData.vNmodes] += \
                                        1j*complexDelta*vFullTimeModes[0,(iparam+1)*romData.vNmodes:(iparam+2)*romData.vNmodes]
                                elif sensInit[iInit]=="zero":
                                    #If finite-diff is zero at t=0 then u(0,x+delta)=u(0,x)
                                    perturbedRomCoeff = romCoeff[[0],:romData.uNmodes+romData.vNmodes]
                                perturbedParams = baseParams.copy()
                                perturbedParams[paramSelect[iparam]]+= complexDelta*1j
                                perturbedModel=TankModel(nCollocation=nCollocation,nElements=nElements,spacing="legendre",bounds=[0,1],params=perturbedParams)
                                dydtPodRom = lambda y,t: perturbedModel.dydtPodRom(y,t,romData,paramSelect = [],penaltyStrength=0)
                                t=0
                                while t<tmax:
                                    odeOut= scipy.integrate.solve_ivp(lambda t,y: dydtPodRom(y,t),(t,t+tstep),perturbedRomCoeff[-1], method='BDF',atol=1e-10,rtol=1e-10)
                                    change = np.max(np.abs(odeOut.y[:,-1]-perturbedRomCoeff[-1,:]))
                                    perturbedRomCoeff=np.append(perturbedRomCoeff,odeOut.y[:,[-1]].transpose(),axis=0)
                                    t+=tstep
                                #Compute Sensitivity in POD space
                                romCoeff = np.append(romCoeff,np.imag(perturbedRomCoeff)/complexDelta,axis=1)

                    
                    #----------------------------- Map Results Back into Spatial Space
                    for i in range(0, neq):
                        # Compute ROM Solution
                        romModeStart = i*(romData.uNmodes+romData.vNmodes)
                        if i==0:
                            uResults[i,:,1,:] = (romData.uModes @ romCoeff[:,romModeStart:romModeStart+romData.uNmodes].transpose()).transpose() + romData.uMean
                            vResults[i,:,1,:] = (romData.vModes @ romCoeff[:,romModeStart+romData.uNmodes:romModeStart+romData.uNmodes+romData.vNmodes].transpose()).transpose() + romData.vMean
                        else:
                            uResults[i,:,1,:] = (romData.uModes @ romCoeff[:,romModeStart:romModeStart+romData.uNmodes].transpose()).transpose()
                            vResults[i,:,1,:] = (romData.vModes @ romCoeff[:,romModeStart+romData.uNmodes:romModeStart+romData.uNmodes+romData.vNmodes].transpose()).transpose()
                        combinedResults[2*i,:,1,:] = uResults[i,:,1,:]
                        combinedResults[2*i+1,:,1,:] = vResults[i,:,1,:]
                        if i==0:
                            uResults[i,:,2,:] = ((romData.uModes @ romData.uTimeModes[:romCoeff.shape[0],:].transpose())+romData.uMean.reshape((romData.uMean.size,1))).transpose()
                            vResults[i,:,2,:] = (romData.vModes @ romData.vTimeModes[:romCoeff.shape[0],:].transpose() +romData.vMean.reshape((romData.vMean.size,1))).transpose()
                        else:
                            #We can generalize POD to the variationin the POD space projected back to the FOM space, regardless of whether sensitivities were in initial POD decomp
                            #Note: Confirmed that, if POD modes are computed using sensitivity snapshots, then this is equivalent to those modes if no mean decomp is used
                            #UNVERIFIED: That that property holds numerically with this implementation and whether it holds if using a mean decomp
                            if mean_reduction[imean]!="zero":
                                print("WARNING: Correctness of approach unconfirmed for non-zero mean reduction")
                            uResults[i,:,2,:] = ((romData.uModes @ romData.uModesWeighted.transpose()) @ uResults[i,:,0,:].transpose()).transpose()
                            vResults[i,:,2,:] = ((romData.vModes @ romData.vModesWeighted.transpose()) @ vResults[i,:,0,:].transpose()).transpose()
                        combinedResults[2*i,:,2,:] = uResults[i,:,2,:]
                        combinedResults[2*i+1,:,2,:] = vResults[i,:,2,:]
                #================================================== Compute Error =================================================================
                    for k in range(len(error_norm)):
                        error[iret,imean,k] = model.computeRomError(uResults[0,:,0,:].transpose(),vResults[0,:,0,:].transpose(),uResults[0,:,1,:].transpose(),vResults[0,:,1,:].transpose(),romData.W,norm=error_norm[k])
                        print("ROM Error in norm "+error_norm[k]+": ", error[iret,imean,k])
                        #INCOMPLETE: Figure out what want to compute for sensitivity error.

                if usePodRom:
                    legends = ["FOM","ROM","POD"] 
                else:
                    legends = ["FOM"]
                #=========================================== Make Movies ===================================================================
                if usePodRom and makeMovies:   
                    subplotMovie([u for u in uResults], x, romSaveFolder + "u.mov", fps=15, xLabels="x", yLabels=uLabels, legends=legends, legendLoc="upper left", subplotSize=(2.5, 2))
                    subplotMovie([v for v in vResults], x, romSaveFolder + "v.mov", fps=15, xLabels="x", yLabels=vLabels, legends=legends, legendLoc="upper left", subplotSize=(2.5, 2))
                    subplotMovie([y for y in combinedResults], x, romSaveFolder + "combined.mov", fps=15, xLabels="x",  yLabels=combinedLabels, legends=legends, legendLoc="upper left", subplotSize=(2.5, 2))
                    if plotError:
                        subplotMovie([u[:,1:3,:]-u[:,[0],:] for u in uResults], x, romSaveFolder + "uError.mov", fps=15, xLabels="x", yLabels=uLabels, legends=legends[1:3], legendLoc="upper left", subplotSize=(2.5, 2),lineTypeStart=1)
                        subplotMovie([v[:,1:3,:]-v[:,[0],:] for v in vResults], x, romSaveFolder + "vError.mov", fps=15, xLabels="x", yLabels=vLabels, legends=legends[1:3], legendLoc="upper left", subplotSize=(2.5, 2),lineTypeStart=1)   
                elif makeMovies:   
                    subplotMovie([u for u in uResults], x, fomSaveFolder + "u.mov", fps=15, xLabels="x", yLabels=uLabels, legends=legends, legendLoc="upper left", subplotSize=(2.5, 2))
                    subplotMovie([v for v in vResults], x, fomSaveFolder + "v.mov", fps=15, xLabels="x", yLabels=vLabels, legends=legends, legendLoc="upper left", subplotSize=(2.5, 2))
                    subplotMovie([y for y in combinedResults], x, fomSaveFolder + "combined.mov", fps=15, xLabels="x",  yLabels=combinedLabels, legends=legends, legendLoc="upper left", subplotSize=(2.5, 2))
                
                #================================================== Make example plots ============================================================
                if plotTimeSeries:
                    tplot = np.linspace(0,tmax/tstep,4,dtype=int)
                    title = ["t=" + str(round(100000*t*tstep)/100000) for t in tplot]

                    fig,axs = subplotTimeSeries([u[tplot,:,:] for u in uResults], x, xLabels="x", yLabels=uLabels, title = title,legends=legends, subplotSize=(2.65, 2))
                    if usePodRom:
                        plt.savefig(romSaveFolder + "uTimeSeries.pdf", format="pdf")
                        plt.savefig(romSaveFolder + "uTimeSeries.png", format="png")
                    else:
                        plt.savefig(fomSaveFolder + "uTimeSeries.pdf", format="pdf")
                        plt.savefig(fomSaveFolder + "uTimeSeries.png", format="png")
                    fig,axs = subplotTimeSeries([v[tplot,:,:] for v in vResults], x, xLabels="x", yLabels=vLabels, title = title,legends=legends, subplotSize=(2.65, 2))
                    if usePodRom:
                        plt.savefig(romSaveFolder + "vTimeSeries.pdf", format="pdf")
                        plt.savefig(romSaveFolder + "vTimeSeries.png", format="png")
                    else:
                        plt.savefig(fomSaveFolder + "vTimeSeries.pdf", format="pdf")
                        plt.savefig(fomSaveFolder + "vTimeSeries.png", format="png")
                    fig,axs = subplotTimeSeries([y[tplot,:,:] for y in combinedResults], x, xLabels="x", yLabels=combinedLabels, title = title,legends=legends, subplotSize=(2.65, 2))
                    if usePodRom:
                        plt.savefig(romSaveFolder + "combinedTimeSeries.pdf", format="pdf")
                        plt.savefig(romSaveFolder + "combinedTimeSeries.png", format="png")
                    else:
                        plt.savefig(fomSaveFolder + "combinedTimeSeries.pdf", format="pdf")
                        plt.savefig(fomSaveFolder + "combinedTimeSeries.png", format="png")
                    if usePodRom and plotError:
                        fig,axs = subplotTimeSeries([u[tplot,1:3,:]-u[tplot,0:1,:] for u in uResults], x, xLabels="x", yLabels=uLabels, title = title,legends=legends[1:3], subplotSize=(2.65, 2),lineTypeStart=1)
                        plt.savefig(romSaveFolder + "uErrorTimeSeries.pdf", format="pdf")
                        plt.savefig(romSaveFolder + "uErrorTimeSeries.png", format="png")
                        fig,axs = subplotTimeSeries([v[tplot,1:3,:]-v[tplot,0:1,:] for v in vResults], x, xLabels="x", yLabels=vLabels, title = title,legends=legends[1:3], subplotSize=(2.65, 2),lineTypeStart=1)
                        plt.savefig(romSaveFolder + "vErrorTimeSeries.pdf", format="pdf")
                        plt.savefig(romSaveFolder + "vErrorTimeSeries.png", format="png")

                #================================================== Plot POD Modes ============================================================
                if plotModes and usePodRom:
                    subplot([mode for mode in romData.uModes.transpose()], x, xLabels="x", yLabels=["Mode " + str(i+1) for i in range(romData.uNmodes)])
                    plt.savefig(romSaveFolder + "uModes.pdf", format="pdf")
                    plt.savefig(romSaveFolder + "uModes.png", format="png")
                    subplot([mode for mode in romData.vModes.transpose()], x, xLabels="x", yLabels=["Mode " + str(i+1) for i in range(romData.vNmodes)])
                    plt.savefig(romSaveFolder + "vModes.pdf", format="pdf")
                    plt.savefig(romSaveFolder + "vModes.png", format="png")
        #=========================================== Plot Convergence ===================================================================
        if usePodRom and plotConvergence and error.size>1:
            if error.shape[2]>1: 
                error = error.reshape((error.shape[0],error.shape[1]*error.shape[2]))
                truncationError = truncationError.reshape((error.shape[0],error.shape[1]))

                if len(mean_reduction)>1:
                    legends = [mean_method +", "+ norm for mean_method in mean_reduction for norm in error_norm]
                else:
                    legends =error_norm
            else:
                legends=mean_reduction
                error=np.squeeze(error)
            fig,axs = plotErrorConvergence(error,truncationError,xLabel="Proportion Information Truncated in POD",yLabel="Relative ROM Error",legends=legends) 
            plt.savefig(romSaveFolder + "../errorConvergence.pdf", format="pdf")
            plt.savefig(romSaveFolder + "../errorConvergence.png", format="png")
plt.show()
