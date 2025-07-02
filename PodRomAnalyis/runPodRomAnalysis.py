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
from postProcessing.plot import plotRomMatrices
from tankModel.TankModel import TankModel
import matplotlib.pyplot as plt


#Set run details
#FOM parameters
paramSet = "BizonChaotic" #BizonPeriodic, BizonLinear, BizonNonLinear, BizonAdvecDiffusion
stabalized=True
equationSet = "tankOnly" #tankOnly, Le, vH, linearParams, linearBoundaryParams, allParams, nonBoundaryParams
nCollocation=1
nElements=128
odeMethod="BDF" #LSODA, BDF, RK45, RK23, DOP853

#romParameters
usePodRom=True
romSensitivityApproach = ["finite","sensEq","complex"] #none, finite, sensEq, DEPOD (unfinished), complex, only used if equationSet!=tankOnly
finiteDelta = 1e-6   #Only used if equationSet!=tankOnly and romSensitivityApproach=="finite"
complexDelta = 1e-10 #Only used if equationSet!=tankOnly and romSensitivityApproach=="complex"
useEnergyThreshold=False
nonlinDim=.4

nPoints=99
nT=200
penaltyStrength=0
sensInit = ["pod","zero"]
quadRule = ["simpson"] # simpson, gauss-legendre, uniform, monte carlo
mean_reduction = ["mean"]
adjustModePairs=False
error_norm = [r"$L_2$",r"$L_\infty$"]

#Display settings
showPlots= True


plotConvergence=True

plotTimeSeries=False
plotModes=False
plotError=False
plotRomCoeff=False
plotSingularValues=False

# plotTimeSeries=False
# plotModes=False
# plotError=False
# plotRomCoeff=False
# plotSingularValues=False

makeMovies=False


if useEnergyThreshold==True:
    #modeRetention=[.85,.885,.925,.95,.975,.99,.999]
    modeRetention=[.85,.99,.999]
    #modeRetention=[.99]
    #modeRetention=[.99,.999]
#Set-up planned convergence ranges to better keep track of stable domains for each case
elif plotConvergence:
    if stabalized:
        if paramSet=="BizonChaotic":
            modeRetention = list(range(6,29))
        elif paramSet=="BizonPeriodic":
            modeRetention = list(range(6,19))
    else: 
        if paramSet=="BizonLinear":
            modeRetention = list(range(1,15))
        elif paramSet== "BizonPeriodic":
            modeRetention = list(range(1,23))
        elif paramSet == "BizonChaotic":
            modeRetention = list(range(1,33))
else:
    modeRetention=6

if stabalized:
    stabalizationTime=150
    tmax=4.1
else:
    tmax=1.5

tPoints= np.linspace(0,tmax,num=nT)
if paramSet == "BizonChaotic":
    baseParams={"PeM": 700, "PeT": 700, "f": .3, "Le": 1, "Da": .15, "beta": 1.8, "gamma": 10,"delta": 2, "vH":-.065}
elif paramSet == "BizonPeriodic":
    baseParams={"PeM": 300, "PeT": 300, "f": .3, "Le": 1, "Da": .15, "beta": 1.4, "gamma": 10,"delta": 2, "vH":-.045}
elif paramSet == "BizonPeriodicReduced":
    baseParams={"PeM": 300, "PeT": 300, "f": .3, "Le": 1, "Da": .08966443, "beta": 1.4, "gamma": 10,"delta": 2, "vH":-.045}
elif paramSet == "BizonLinear":
    baseParams={"PeM": 300, "PeT": 300, "f": .3, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 2, "vH":-.045}
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
    stabalizationTime=.1
    tmax=1.5
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
    raise ValueError("Invalid paramSet entered: " + str(equationSet))

fomSaveFolder = "../../results/podRomAnalysis/"+paramSet
if not stabalized:
    fomSaveFolder+="Unstabalized"    
fomSaveFolder += "_nCol" + str(nCollocation) + "_nElem"+str(nElements) + "_nT" + str(nT) +"/"+equationSet
        


#Simulation Settings
bounds = [0,1]
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
model=TankModel(nCollocation=nCollocation,nElements=nElements,spacing="legendre",bounds=bounds,params=baseParams)
dydtSens =lambda y,t: model.dydtSens(y,t,paramSelect=paramSelect)

period = 1
init = lambda x,b: b[0]+b[1]/bounds[1]*x+b[2]*np.cos(2*np.pi*x*period/bounds[1])+b[3]*np.sin(2*np.pi*x*period/bounds[1])
uCoeff = np.empty((4,))
uCoeff[0]=0
uCoeff[1]=1
uCoeff[2]=-uCoeff[0]
uCoeff[3]=-uCoeff[1]/(2*np.pi*period)

vCoeff = np.empty((4,))
vCoeff[0]=.35
vCoeff[1]=.2
vCoeff[2]=-vCoeff[0]+(baseParams["f"]*vCoeff[1])/(1-baseParams["f"])
vCoeff[3]=-vCoeff[1]/(2*np.pi*period)
modelCoeff=np.append(init(model.collocationPoints,uCoeff),init(model.collocationPoints,vCoeff),axis=0)
for i in range(neq-1):
    modelCoeff = np.append(modelCoeff,[0*model.collocationPoints],axis=0)
    modelCoeff = np.append(modelCoeff,[3/2*model.collocationPoints**3-model.collocationPoints**3],axis=0)

#=================================== Run Stabalization ===========================================================================
print("Running Stabalization")
if stabalized:
    #Run out till stabalizing in periodic domain
    odeOut= scipy.integrate.solve_ivp(lambda t,y: dydtSens(y,t),(0,stabalizationTime),modelCoeff, method=odeMethod,atol=1e-6,rtol=1e-6)
    print(odeOut.y.shape)
    modelCoeff = odeOut.y[:,-1].transpose()
#=================================== Get Simulation Data ================================================================
print("Getting Simulation Data")
odeOut= scipy.integrate.solve_ivp(lambda t,y: dydtSens(y,t),(0,tmax), modelCoeff, t_eval = tPoints, method=odeMethod,atol=1e-10,rtol=1e-10)
modelCoeff=odeOut.y.transpose()
print(modelCoeff.shape)
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


#================================== Run POD-ROM ==================================================================================
 
for iquad in range(len(quadRule)):
    print("Using ", quadRule[iquad], " quadrature rule")
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
        print("     Using ", romSensitivityApproach[isens], " sensitivity approach")
        for iInit in range(len(sensInit)):
            print("         Using ", sensInit[iInit], " sensitivity initialization")
            for imean in range(len(mean_reduction)):
                error = np.empty((len(modeRetention),len(error_norm)))
                truncationError=np.empty((len(modeRetention),len(error_norm)))
                for iret in range(len(modeRetention)):
                    if usePodRom:
                        print("             Running POD-ROM for mode retention ", modeRetention[iret], " and mean reduction ", mean_reduction[imean])
                        #------------------------------- Make Folder to save data
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
                        
                        if adjustModePairs ==True:
                            podSaveFolder+= "_adjusted"

                        if nonlinDim == "max":
                            podSaveFolder +="/"
                        else:
                            podSaveFolder +="_nDim" + str(nonlinDim) + "/"
                        if useEnergyThreshold:
                            romSaveFolder = podSaveFolder + "e"+str(modeRetention[iret])
                        else :
                            romSaveFolder = podSaveFolder + "m"+str(modeRetention[iret])

                        if nonlinDim == "max":
                            romSaveFolder +="/"
                        else:
                            romSaveFolder +="_nDim" + str(nonlinDim) + "/"

                        if not os.path.exists(podSaveFolder):
                            os.makedirs(podSaveFolder)
                        if not os.path.exists(romSaveFolder) and (plotTimeSeries or plotModes or plotError or plotRomCoeff or plotSingularValues):
                            os.makedirs(romSaveFolder)
                        if iret ==0 and plotSingularValues:
                            romData, null =model.constructPodRom(modelCoeff[:,:2*nCollocation*nElements],x,W,min(nT,nPoints),mean=mean_reduction[imean],useEnergyThreshold=False)
                            fig, axes = plt.subplots(1,1, figsize=(5,4))
                            axes.loglog(romData.uSingularValues,"-b",lw=5,ms=8)
                            axes.loglog(romData.vSingularValues,"--m",lw=5,ms=8)
                            axes.legend(["u","v"])
                            axes.set_xlabel("Mode")
                            axes.set_ylabel("Singular Value")
                            plt.tight_layout()
                            plt.savefig(romSaveFolder + "../singularValues.pdf", format="pdf")
                            plt.savefig(romSaveFolder + "../singularValues.png", format="png")

                        #------------------------------- Compute POD
                        # Get POD Decomposition
                        print("                 Computing POD")
                        if romSensitivityApproach[isens] == "DEPOD" and equationSet == "tankOnly":
                            # INCOMPLETE: need to setup POD computation for mixed-solution and sensitivity snapshot matrices
                            raise ValueError("DEPOD not yet implemented")
                            romData, truncationError[iret,:]=model.constructPodRom(modelCoeff,x,modeRetention[iret],mean=mean_reduction[imean],useEnergyThreshold=useEnergyThreshold)
                        else :
                            romData, truncationError[iret,:]=model.constructPodRom(modelCoeff[:,:2*nCollocation*nElements],x,W,modeRetention[iret],mean=mean_reduction[imean],useEnergyThreshold=useEnergyThreshold,adjustModePairs=adjustModePairs,nonlinDim=nonlinDim)
                        if np.isclose(uFomResult,vFomResult).all():
                            if not np.isclose(romData.uModes,romData.vModes).all():
                                print("Warning: u and v identical but POD modes don't match")
                                # print(np.sum(romData.uModes-romData.vModes,axis=0))
                                print("Integral of u and v POD mode difference:")
                                print(romData.uModesInt-romData.vModesInt)
                         #Compute time modes for sensitivity equations
                        if romSensitivityApproach[isens] != "DEPOD" and equationSet != "tankOnly":
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
                        #Get Initial Modal Values
                        if romSensitivityApproach[isens] == "DEPOD" and equationSet != "tankOnly":
                            # INCOMPLETE: need to setup POD computation for mixed-solution and sensitivity snapshot matrices
                            raise ValueError("DEPOD not yet implemented")
                            #Compute Time modes for sensitivity equations
                            for i in range(1, neq):
                                start = i*(2*model.nCollocation*model.nElements)
                                timeModeIndex = modelCoeff.shape[0]*i
                                romInit[start:start+romData.uNmodes]\
                                    =romData.uTimeModes[timeModeIndex,:romData.uNmodes]
                                romInit[start+romData.uNmodes:start+romData.uNmodes+romData.vNmodes]\
                                    =romData.vTimeModes[timeModeIndex,:romData.vNmodes]
                            dydtPodRom = lambda y,t: model.dydtPodRom(y,t,romData,paramSelect=paramSelect,penaltyStrength=0)
                        else: 
                            romInit=np.empty((romData.uNmodes+romData.vNmodes))
                            romInit[:romData.uNmodes]\
                                =romData.uTimeModes[0,:romData.uNmodes]
                            romInit[romData.uNmodes:romData.uNmodes+romData.vNmodes]\
                                =romData.vTimeModes[0,:romData.vNmodes]
                            dydtPodRom = lambda y,t: model.dydtPodRom(y,t,romData,paramSelect = [],penaltyStrength=penaltyStrength)
                        #Compute Base Rom Value
                        print("nNonlinDim: ", romData.vNonlinDim)
                        odeOut = scipy.integrate.solve_ivp(lambda t,y: dydtPodRom(y,t),(0,tmax),romInit, t_eval=tPoints, method=odeMethod,atol=1e-10,rtol=1e-10)
                        romCoeff = np.empty((modelCoeff.shape[0],neq*(romData.uNmodes+romData.vNmodes)))
                        romCoeff[:,:romData.uNmodes+romData.vNmodes] = odeOut.y.transpose()
                        
                        #------------------------------- Compute Sensitivity
                        if equationSet!="tankOnly":
                            if romSensitivityApproach[isens] == "finite":
                                for iparam in range(len(paramSelect)):
                                    print("Computing sensitivity for " + paramSelect[iparam])
                                    if sensInit[iInit]=="pod":
                                        #If finite-diff is exact at t=0 then u(0,x+delta)=u(0,x)+delta*u'(0,x)
                                        perturbedRomCoeff = romCoeff[0,:romData.uNmodes+romData.vNmodes]
                                        perturbedRomCoeff[:romData.uNmodes] +=\
                                            finiteDelta*uFullTimeModes[0,(iparam+1)*romData.uNmodes:(iparam+2)*romData.uNmodes]
                                        perturbedRomCoeff[romData.uNmodes:romData.uNmodes+romData.vNmodes] += \
                                            finiteDelta*vFullTimeModes[0,(iparam+1)*romData.vNmodes:(iparam+2)*romData.vNmodes]
                                    elif sensInit[iInit]=="zero":
                                        #If finite-diff is zero at t=0 then u(0,x+delta)=u(0,x)
                                        perturbedRomCoeff = romCoeff[0,:romData.uNmodes+romData.vNmodes]
                                    perturbedParams = baseParams.copy()
                                    perturbedParams[paramSelect[iparam]]+= finiteDelta
                                    perturbedModel=TankModel(nCollocation=nCollocation,nElements=nElements,spacing="legendre",bounds=bounds,params=perturbedParams)
                                    dydtPodRom = lambda y,t: perturbedModel.dydtPodRom(y,t,romData,paramSelect = [],penaltyStrength=penaltyStrength)
                                    odeOut= scipy.integrate.solve_ivp(lambda t,y: dydtPodRom(y,t),(0,tmax),perturbedRomCoeff, t_eval = tPoints, method=odeMethod,atol=1e-10,rtol=1e-10)
                                    #Compute Sensitivity in POD space
                                    romCoeff[:,(iparam+1)*(romData.uNmodes+romData.vNmodes):(iparam+2)*(romData.uNmodes+romData.vNmodes)]=\
                                        odeOut.y.transpose()-romCoeff[:,:romData.uNmodes+romData.vNmodes]
                            elif romSensitivityApproach[isens] == "complex":
                                for iparam in range(len(paramSelect)):
                                    print("Computing sensitivity for " + paramSelect[iparam])
                                    #Initialize sensitivity
                                    if sensInit[iInit]=="pod":
                                        #If finite-diff is exact at t=0 then u(0,x+delta)=u(0,x)+delta*u'(0,x)
                                        perturbedRomCoeff = romCoeff[0,:romData.uNmodes+romData.vNmodes].astype(complex)
                                        perturbedRomCoeff[:romData.uNmodes] += \
                                            1j*complexDelta*uFullTimeModes[0,(iparam+1)*romData.uNmodes:(iparam+2)*romData.uNmodes]
                                        perturbedRomCoeff[romData.uNmodes:romData.uNmodes+romData.vNmodes] += \
                                            1j*complexDelta*vFullTimeModes[0,(iparam+1)*romData.vNmodes:(iparam+2)*romData.vNmodes]
                                    elif sensInit[iInit]=="zero":
                                        #If finite-diff is zero at t=0 then u(0,x+delta)=u(0,x)
                                        perturbedRomCoeff = romCoeff[0,:romData.uNmodes+romData.vNmodes].astype(complex)
                                    perturbedParams = baseParams.copy()
                                    perturbedParams[paramSelect[iparam]]+= complexDelta*1j
                                    perturbedModel=TankModel(nCollocation=nCollocation,nElements=nElements,spacing="legendre",bounds=bounds,params=perturbedParams)
                                    dydtPodRom = lambda y,t: perturbedModel.dydtPodRom(y,t,romData,paramSelect = [],penaltyStrength=penaltyStrength)
                                    odeOut= scipy.integrate.solve_ivp(lambda t,y: dydtPodRom(y,t),(0,tmax), perturbedRomCoeff, t_eval = tPoints, method=odeMethod,atol=1e-10,rtol=1e-10)
                                        
                                    #Compute Sensitivity in POD space
                                    romCoeff[:,(iparam+1)*(romData.uNmodes+romData.vNmodes):(iparam+2)*(romData.uNmodes+romData.vNmodes)]=\
                                        np.imag(odeOut.y.transpose())/complexDelta
                            elif romSensitivityApproach[isens] == "sensEq":
                                for iparam in range(len(paramSelect)):
                                    print("Computing sensitivity for " + paramSelect[iparam])
                                    romInit = np.empty((2*(romData.uNmodes+romData.vNmodes)))
                                    romInit[:romData.uNmodes+romData.vNmodes] = romCoeff[0,:romData.uNmodes+romData.vNmodes].copy() 
                                    if sensInit[iInit]=="pod":
                                        romInit[romData.uNmodes+romData.vNmodes:2*romData.uNmodes+romData.vNmodes]\
                                            =uFullTimeModes[0,(iparam+1)*romData.uNmodes:(iparam+2)*romData.uNmodes]
                                        romInit[2*romData.uNmodes+romData.vNmodes:]\
                                            =vFullTimeModes[0,(iparam+1)*romData.vNmodes:(iparam+2)*romData.vNmodes]
                                    elif sensInit[iInit]=="zero":
                                        romInit[romData.uNmodes+romData.vNmodes:] =np.zeros((romData.uNmodes+romData.vNmodes))
                                    else:
                                        raise ValueError("Invalid sensInit entered: " + str(sensInit[iInit]))
                                    dydtPodRom = lambda y,t: model.dydtPodRom(y,t,romData,paramSelect = paramSelect[iparam],penaltyStrength=penaltyStrength)
                                    odeOut= scipy.integrate.solve_ivp(lambda t,y: dydtPodRom(y,t),(0,tmax), romInit, t_eval = tPoints, method=odeMethod,atol=1e-10,rtol=1e-10)
                                        
                                    #Compute Sensitivity in POD space
                                    romCoeff[:,(iparam+1)*(romData.uNmodes+romData.vNmodes):(iparam+2)*(romData.uNmodes+romData.vNmodes)]=\
                                        odeOut.y.transpose()[:,romData.uNmodes+romData.vNmodes:].copy()


            
                    #----------------------------- Map Results Back into Spatial Space
                    for i in range(0, neq):
                        # Compute ROM Solution
                        romModeStart = i*(romData.uNmodes+romData.vNmodes)
                        #ROM Error
                        if i==0:
                            uResults[i,:,1,:] = (romData.uModes @ romCoeff[:,romModeStart:romModeStart+romData.uNmodes].transpose()).transpose() + romData.uMean
                            vResults[i,:,1,:] = (romData.vModes @ romCoeff[:,romModeStart+romData.uNmodes:romModeStart+romData.uNmodes+romData.vNmodes].transpose()).transpose() + romData.vMean
                        else:
                            uResults[i,:,1,:] = (romData.uModes @ romCoeff[:,romModeStart:romModeStart+romData.uNmodes].transpose()).transpose()
                            vResults[i,:,1,:] = (romData.vModes @ romCoeff[:,romModeStart+romData.uNmodes:romModeStart+romData.uNmodes+romData.vNmodes].transpose()).transpose()
                        combinedResults[2*i,:,1,:] = uResults[i,:,1,:]
                        combinedResults[2*i+1,:,1,:] = vResults[i,:,1,:]
                        #POD Error
                        if i==0:
                            uResults[i,:,2,:] = ((romData.uModes @ romData.uTimeModes[:romCoeff.shape[0],:].transpose())+romData.uMean.reshape((romData.uMean.size,1))).transpose()
                            vResults[i,:,2,:] = (romData.vModes @ romData.vTimeModes[:romCoeff.shape[0],:].transpose() +romData.vMean.reshape((romData.vMean.size,1))).transpose()
                        else:
                            #We can generalize POD to the variationin the POD space projected back to the FOM space, regardless of whether sensitivities were in initial POD decomp
                            #Note: Confirmed that, if POD modes are computed using sensitivity snapshots, then this is equivalent to those modes if no mean decomp is used
                            #UNVERIFIED: That that property holds numerically with this implementation and whether it holds if using a mean decomp
                            if mean_reduction[imean]!="zero":
                                print("WARNING: Correctness of approach unconfirmed for non-zero mean reduction")
                            
                            uResults[i,:,2,:] = (romData.uModes @ uFullTimeModes[:,i*romData.uNmodes:(i+1)*romData.uNmodes].transpose()+romData.uMean.reshape((romData.uMean.size,1))).transpose()
                            vResults[i,:,2,:] = (romData.vModes @ vFullTimeModes[:,i*romData.vNmodes:(i+1)*romData.vNmodes].transpose()+romData.vMean.reshape((romData.vMean.size,1))).transpose()
                        combinedResults[2*i,:,2,:] = uResults[i,:,2,:]
                        combinedResults[2*i+1,:,2,:] = vResults[i,:,2,:]
                #================================================== Compute Error =================================================================
                    for k in range(len(error_norm)):
                        error[iret,k] = model.computeRomError(uResults[0,:,0,:].transpose(),vResults[0,:,0,:].transpose(),uResults[0,:,1,:].transpose(),vResults[0,:,1,:].transpose(),romData.W,tPoints=tPoints,norm=error_norm[k])
                        print(                  "ROM Error in norm "+error_norm[k]+": ", error[iret,k])
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
                            subplotMovie([u[:,1:3,:]-u[:,[0],:] for u in uResults], x, romSaveFolder + "uError.mov", fps=15, xLabels="x", yLabels=uLabels, legends=legends[1:3], legendLoc="upper left", subplotSize=(2.5, 2),lineTypeStart=1,yRanges="auto")
                            subplotMovie([v[:,1:3,:]-v[:,[0],:] for v in vResults], x, romSaveFolder + "vError.mov", fps=15, xLabels="x", yLabels=vLabels, legends=legends[1:3], legendLoc="upper left", subplotSize=(2.5, 2),lineTypeStart=1,yRanges="auto")   
                    elif makeMovies:   
                        subplotMovie([u for u in uResults], x, fomSaveFolder + "u.mov", fps=15, xLabels="x", yLabels=uLabels, legends=legends, legendLoc="upper left", subplotSize=(2.5, 2))
                        subplotMovie([v for v in vResults], x, fomSaveFolder + "v.mov", fps=15, xLabels="x", yLabels=vLabels, legends=legends, legendLoc="upper left", subplotSize=(2.5, 2))
                        subplotMovie([y for y in combinedResults], x, fomSaveFolder + "combined.mov", fps=15, xLabels="x",  yLabels=combinedLabels, legends=legends, legendLoc="upper left", subplotSize=(2.5, 2))
                    
                    #================================================== Make example plots ============================================================
                    if plotTimeSeries:
                        tplot = np.linspace(0,tPoints.size-1,4,dtype=int)
                        title = ["t=" + str(round(1000*tPoints[it])/1000) for it in tplot]

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
                        if plotRomCoeff:
                            coeffLabels = ["Coeff " + str(i+1) for i in range(romData.uNmodes)]
                            uCoeffData=[np.array([rom,pod]) for pod,rom in zip(romData.uTimeModes[:,:romData.uNmodes].transpose(),\
                                                                               romCoeff[:,:romData.uNmodes].transpose())]
                            
                            fig,axs = subplot(uCoeffData, tPoints, xLabels="t", yLabels=coeffLabels, legends=legends[1:3], subplotSize=(2.65, 2),lineTypeStart=1)
                            plt.savefig(romSaveFolder + "uRomCoeff.pdf", format="pdf")
                            plt.savefig(romSaveFolder + "uRomCoeff.png", format="png")

                            coeffLabels = ["Coeff " + str(i+1) for i in range(romData.vNmodes)]
                            vCoeffData=[np.array([rom,pod]) for pod,rom in zip(romData.vTimeModes[:,:romData.vNmodes].transpose(),\
                                                                               romCoeff[:,romData.uNmodes:romData.uNmodes+romData.vNmodes].transpose())]
                            fig,axs = subplot(vCoeffData, tPoints, xLabels="t", yLabels=coeffLabels,legends=legends[1:3], subplotSize=(2.65, 2),lineTypeStart=1)
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
                                    print(adjustedV-baseV)
                                    print(nonLinearTerm(baseU,baseV))
                                    print((nonLinearTerm(baseU,adjustedV)-nonLinearTerm(baseU,baseV))/1e-6)
                                    nonLinearLinearized[it][i,:]=(nonLinearTerm(baseU,adjustedV)-nonLinearTerm(baseU,baseV))/1e-6
                            
                            fig, axes = plotRomMatrices(nonLinearLinearized,\
                                                        xLabels=r"$v_j$",yLabels=[r"$\frac{\partial \mathcal{N}(v_i)}{\partial v_j}$"," "," ", " "],\
                                                        title=["t=" + str(round(1000*tPoints[it])/1000) for it in tplot],\
                                                        cmap="coolwarm")
                            plt.savefig(romSaveFolder + "vRomMatrices_linearization.pdf", format="pdf")
                            plt.savefig(romSaveFolder + "vRomMatrices_linearization.png", format="png")

                    #================================================== Plot POD Modes ============================================================
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
                        

                    #================================================== Plot Singular Values =======================================================
                    if plotSingularValues:
                        fig, axes = plt.subplots(1,1, figsize=(5,4))
                        axes.semilogy(np.arange(1,romData.uSingularValues.size+1),romData.uSingularValues,"bs",lw=5,ms=8)
                        axes.semilogy(np.arange(1,romData.vSingularValues.size+1),romData.vSingularValues,"mo",lw=5,ms=8)
                        axes.legend(["u","v"])
                        axes.set_xlabel("Mode")
                        axes.set_ylabel("Singular Value")
                        plt.tight_layout()
                        plt.savefig(romSaveFolder + "singularValues.pdf", format="pdf")
                        plt.savefig(romSaveFolder + "singularValues.png", format="png")
                    if not showPlots:
                        plt.close()
                #=========================================== Plot Convergence ===================================================================
                if usePodRom and plotConvergence and error.size>1:
                    if error.ndim>2: 
                        error = error.reshape((error.shape[0],error.shape[1]*error.shape[2]))
                        truncationError = truncationError.reshape((error.shape[0],error.shape[1]))

                        if len(mean_reduction)>1:
                            legends = [mean_method +", "+ norm for mean_method in mean_reduction for norm in error_norm]
                        else:
                            legends =error_norm
                    else:
                        legends=error_norm
                        error=np.squeeze(error)
                    fig,axs = plotErrorConvergence(error,truncationError,xLabel="Proportion Information Truncated in POD",yLabel="Relative ROM Error",legends=legends) 
                    plt.savefig(podSaveFolder + "errorConvergence_s"+str(modeRetention[0])+"_e" + str(modeRetention[-1])+".pdf", format="pdf")
                    plt.savefig(podSaveFolder + "errorConvergence_s"+str(modeRetention[0])+"_e" + str(modeRetention[-1])+".png", format="png")
if showPlots:
    plt.show()
