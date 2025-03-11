import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath(os.path.join(current_script_dir,'..',))
sys.path.append(grandparent_dir)
print(sys.path)
import numpy as np
import scipy
from postProcessing.plot import subplotTimeSeries
from postProcessing.plot import subplotMovie
from postProcessing.plot import plotErrorConvergence
from tankModel.TankModel import TankModel
import matplotlib.pyplot as plt


#Set save details
paramSet = "BizonAdvec" #BizonPeriodic, BizonLinear, BizonAdvecDiffusion
equationSet = "tankOnly"
romSensitivityApproach = "finite" #none, finite, sensEq, DEPOD 
nCollocation=2
nElements=64
usePodRom=True
useEnergyThreshold=False
if useEnergyThreshold==True:
    #energyRetention=[.8,.85,.885,.925,.95,.975,.99,.9925,.995,.9975,.999,.9999]
    modeRetention=[.85,.99,.999]
else:
    #modeRetention=[3,4]
    #modeRetention = [3,4,5,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23] #
    modeRetention = [1,3,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
    #modeRetention = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24] 
xpoints=101
mean_reduction = ["zero"]
error_norm = [r"$L_2$",r"$L_\infty$"]

#Display settings
plotTimeSeries=False
plotError=True
makeMovies=False



if paramSet == "BizonChaotic":
    baseParams={"PeM": 700, "PeT": 700, "f": .3, "Le": 1, "Da": .15, "beta": 1.8, "gamma": 10,"delta": 2, "vH":-.065}
    stabalizationTime=150
    tstep=.02
    tmax=4.1
elif paramSet == "BizonPeriodic":
    baseParams={"PeM": 300, "PeT": 300, "f": .3, "Le": 1, "Da": .15, "beta": 1.4, "gamma": 10,"delta": 2, "vH":-.045}
    stabalizationTime=100
    tstep=.02
    tmax=4.1
elif paramSet == "BizonNonLinear":
    baseParams={"PeM": 300, "PeT": 300, "f": .3, "Le": 1, "Da": .15, "beta": 1.4, "gamma": 10,"delta": 0, "vH":0}
    stabalizationTime=.1
    tstep=.02
    tmax=4.1
elif paramSet == "BizonLinear":
    baseParams={"PeM": 300, "PeT": 300, "f": 0, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 2, "vH":-.045}
    stabalizationTime=.1
    tstep=.02
    tmax=2
elif paramSet == "BizonAdvecDiffusion":
    baseParams={"PeM": 1, "PeT": 1, "f": 0, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 0, "vH":0}
    stabalizationTime=.1
    tstep=.01
    tmax=1.5
elif paramSet == "BizonAdvec":
    baseParams={"PeM": 1e16, "PeT": 1e16, "f": 0, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 0, "vH":0}
    stabalizationTime=.1
    tstep=.01
    tmax=1
else: 
    raise ValueError("Invalid paramSet entered: " + str(equationSet))
    


fomSaveFolder = "../../results/podRomAnalysis/"+paramSet +"_"+equationSet +"_nCol" + str(nCollocation) + "_nElem"+str(nElements)


#Simulation Settings
#Determine parameters to get sensitivity of
if equationSet == "tankOnly":
    neq=1
    paramSelect=[]
    uLabels=[r"$u$"]
    vLabels=[r"$v$"]
    combinedLabels= [r"$u$",r"$v$"]
elif equationSet == "linearParams":
    neq=5
    paramSelect=["Le","Da","delta","vH"]
    uLabels=[r"$u$",r"$u_{\mathrm{Le}}$",r"$u_{\mathrm{Da}}$",r"$u_{\delta}$",r"$u_{v_H}$"]
    vLabels=[r"$v$",r"$v_{\mathrm{Le}}$",r"$v_{\mathrm{Da}}$",r"$v_{\delta}$",r"$v_{v_H}$"]
    combinedLabels= [r"$u$",r"$u_{\mathrm{Le}}$",r"$u_{\mathrm{Da}}$",r"$u_{\delta}$",r"$u_{v_H}$",r"$v$",r"$v_{\mathrm{Le}}$",r"$v_{\mathrm{Da}}$",r"$v_{\delta}$",r"$v_{v_H}$"]
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
else:
    raise ValueError("Invalid equationSet entered: " + str(equationSet))
if not os.path.exists(fomSaveFolder):
        os.makedirs(fomSaveFolder)
#==================================== Setup system ===============================================================================
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
    if equationSet != "tankOnly":
        print("WARNING: Initialization of advection diffusion case not yet implemented for non-tankOnly equation sets")
    collPoints=np.append(model.collocationPoints,model.collocationPoints,axis=0)
    modelCoeff=np.array([-collPoints**2+2*collPoints])
elif baseParams["vH"]==0:
    modelCoeff=np.ones((1,model.nCollocation*model.nElements*2*neq))*.5
else:
    modelCoeff=np.ones((1,model.nCollocation*model.nElements*2*neq))*baseParams["vH"]
x=np.linspace(0,1,xpoints)
#=================================== Run Stabalization ===========================================================================
if not stabalizationTime==0:
    #Run out till stabalizing in periodic domain
    odeOut= scipy.integrate.solve_ivp(lambda t,y: dydtSens(y,t),(0,stabalizationTime),modelCoeff[-1,:], method='BDF',atol=1e-6,rtol=1e-6)
    modelCoeff = odeOut.y[:,[-1]].transpose()
#=================================== Get Simulation Data ================================================================
t=0
while t<tmax:
    odeOut= scipy.integrate.solve_ivp(lambda t,y: dydtSens(y,t),(t,t+tstep),modelCoeff[-1], method='BDF',atol=1e-10,rtol=1e-10)
    change = np.max(np.abs(odeOut.y[:,-1]-modelCoeff[-1,:]))
    modelCoeff=np.append(modelCoeff,odeOut.y[:,[-1]].transpose(),axis=0)
    #print("(t,change): (", t,", ", change, ")")
    t+=tstep
#pre-allocation results storage
if usePodRom:
    uResults = np.empty((neq,modelCoeff.shape[0],3,x.size))
    vResults = np.empty((neq,modelCoeff.shape[0],3,x.size))
    combinedResults=np.empty((2*neq,modelCoeff.shape[0],3,x.size))
else:
    uResults = np.empty((neq,modelCoeff.shape[0],x.size))
    vResults = np.empty((neq,modelCoeff.shape[0],x.size))
    combinedResults=np.empty((2*neq,modelCoeff.shape[0],x.size))
if type(modeRetention)==float or type(modeRetention)==int:
    modeRetention = [modeRetention]
if type(mean_reduction)==str:
    mean_reduction = [mean_reduction]
error = np.empty((len(modeRetention),len(mean_reduction),len(error_norm)))
truncationError=np.empty((len(modeRetention),len(mean_reduction),len(error_norm)))
#--------------------------------- Map results back to spatial points
for i in range(0, neq):
    fomStart = i*(2*model.nCollocation*model.nElements)
    uFomResult, vFomResult = model.eval(x,modelCoeff[:,fomStart:fomStart+2*model.nCollocation*model.nElements],output="seperated")
    uResults[i,:,0,:]=uFomResult
    vResults[i,:,0,:]=vFomResult
    combinedResults[2*i,:,0,:]=uFomResult
    combinedResults[2*i+1,:,0,:]=vFomResult
#================================== Run POD-ROM ==================================================================================
for iret in range(len(modeRetention)):
    for j in range(len(mean_reduction)):
        if usePodRom:
            if j>1:
                raise ValueError("More than 1 mean-reduction used. Results data currently gets saved over for each mean-reduction")
            #------------------------------- Make Folder to save data
            if plotTimeSeries or makeMovies:
                if useEnergyThreshold:
                    romSaveFolder = fomSaveFolder + "/podRom_e"+str(modeRetention[iret])+"_"+mean_reduction[j]+ "/"
                else :
                    romSaveFolder = fomSaveFolder + "/podRom_n"+str(modeRetention[iret])+"_"+mean_reduction[j]+ "/"

                if not os.path.exists(romSaveFolder):
                    os.makedirs(romSaveFolder)

            #------------------------------- Compute POD
            # Get POD Decomposition
            if romSensitivityApproach == "finite" or romSensitivityApproach == "none":
                #No adjustment to POD Needed for finite difference approach
                romData, truncationError[iret,j,:]=model.constructPodRom(modelCoeff[:,:2*nCollocation*nElements],x,modeRetention[iret],mean=mean_reduction[j],useEnergyThreshold=useEnergyThreshold)
            elif romSensitivityApproach == "DEPOD":
                # INCOMPLETE: need to setup POD computation for mixed-solution and sensitivity snapshot matrices
                raise ValueError("DEPOD not yet implemented")
                romData, truncationError[iret,j,:]=model.constructPodRom(modelCoeff,x,modeRetention[iret],mean=mean_reduction[j],useEnergyThreshold=useEnergyThreshold)
            elif romSensitivityApproach == "sensEq":
                romData, truncationError[iret,j,:]=model.constructPodRom(modelCoeff[:,:2*nCollocation*nElements],x,modeRetention[iret],mean=mean_reduction[j],useEnergyThreshold=useEnergyThreshold)
                #Compute time modes for sensitivity equations
                uFullSnapshots,vFullSnapshots = model.eval(x,modelCoeff,output="combined")
                uFullTimeModes = romData.uModes.transpose() @ uFullSnapshots
                vFullTimeModes = romData.vModes.transpose() @ vFullSnapshots
                #Check Time modes match for first nModes
                if not np.isclose(uFullTimeModes[:,:nCollocation*nElements],romData.uTimeModes).all():
                    raise ValueError("u time modes do not match")
                if not np.isclose(uFullTimeModes[:,:nCollocation*nElements],romData.uTimeModes).all():
                    raise ValueError("v time modes do not match")
            else:
                raise ValueError("Invalid romSensitivityApproach entered: " + str(romSensitivityApproach))
            print("Number of POD Modes for (u,v): (",romData.uNmodes, ", ",romData.vNmodes, ")")


            #------------------------------ Run POD-ROM
            #Get Initial Modal Weights
            if romSensitivityApproach == "finite" or romSensitivityApproach == "none":
                romCoeff=np.empty((1,romData.uNmodes+romData.vNmodes))
                romCoeff[0,:romData.uNmodes]\
                    =romData.uTimeModes[0,:romData.uNmodes]
                romCoeff[0,romData.uNmodes:romData.uNmodes+romData.vNmodes]\
                    =romData.vTimeModes[0,:romData.vNmodes]
            elif romSensitivityApproach == "DEPOD":
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
            elif romSensitivityApproach == "sensEq":
                romCoeff=np.empty((1,neq*(romData.uNmodes+romData.vNmodes)))
                for ieq in range(neq):
                    start = ieq*romData.uNmodes+romData.vNmodes
                    romCoeff[0,start:start+romData.uNmodes]\
                        =uFullTimeModes[0,ieq*romData.uNmodes:(ieq+1)*romData.uNmodes]
                    romCoeff[0,start+romData.uNmodes:start+romData.uNmodes+romData.vNmodes]\
                        =vFullTimeModes[0,ieq*romData.vNmodes:(ieq+1)*romData.vNmodes]
            dydtPodRom = lambda y,t: model.dydtPodRom(y,t,romData,paramSelect=paramSelect,penaltyStrength=0)
            t=0
            while t<tmax:
                odeOut= scipy.integrate.solve_ivp(lambda t,y: dydtPodRom(y,t),(t,t+tstep),romCoeff[-1], method='BDF',atol=1e-10,rtol=1e-10)
                change = np.max(np.abs(odeOut.y[:,-1]-romCoeff[-1,:]))
                romCoeff=np.append(romCoeff,odeOut.y[:,[-1]].transpose(),axis=0)
                t+=tstep
            if romSensitivityApproach == "finite":
                perturbedRomCoeff = romCoeff[[0],:]
                for i in range(len(paramSelect)):
                    perturbedParams = baseParams.copy()
                    perturbedParams[paramSelect[i]]+= finiteDelta
                    model.params = perturbedParams
                    while t<tmax:
                        odeOut= scipy.integrate.solve_ivp(lambda t,y: dydtPodRom(y,t),(t,t+tstep),perturbedRomCoeff[-1], method='BDF',atol=1e-10,rtol=1e-10)
                        change = np.max(np.abs(odeOut.y[:,-1]-perturbedRomCoeff[-1,:]))
                        pertrurbedRomCoeff=np.append(romCoeff,odeOut.y[:,[-1]].transpose(),axis=0)
                        t+=tstep
                    romCoeff = np.append(romCoeff,(perturbedRomCoeff-romCoeff[:,:romData.uNmodes+romData.vNmodes])/finiteDelta,axis=0)
            
            #----------------------------- Map Results Back into Spatial Space
            for i in range(0, neq):
                # Compute ROM Solution
                romStart = i*(romData.uNmodes+romData.vNmodes)
                if i==0:
                    uResults[i,:,1,:] = (romData.uModes @ romCoeff[:,romStart:romStart+romData.uNmodes].transpose()).transpose() + romData.uMean
                    vResults[i,:,1,:] = (romData.vModes @ romCoeff[:,romStart+romData.uNmodes:romStart+romData.uNmodes+romData.vNmodes].transpose()).transpose() + romData.vMean
                else:
                    uResults[i,:,1,:] = (romData.uModes @ romCoeff[:,romStart:romStart+romData.uNmodes].transpose()).transpose()
                    vResults[i,:,1,:] = (romData.vModes @ romCoeff[:,romStart+romData.uNmodes:romStart+romData.uNmodes+romData.vNmodes].transpose()).transpose()
                
                if i==0:
                    uResults[i,:,2,:] = ((romData.uModes @ romData.uTimeModes[:romCoeff.shape[0],:].transpose())+romData.uMean.reshape((romData.uMean.size,1))).transpose()
                    vResults[i,:,2,:] = (romData.vModes @ romData.vTimeModes[:romCoeff.shape[0],:].transpose() +romData.vMean.reshape((romData.vMean.size,1))).transpose()
                else:
                    #We can generalize POD to the variationin the POD space projected back to the FOM space, regardless of whether sensitivities were in initial POD decomp
                    #Note: Confirmed that, if POD modes are computed using sensitivity snapshots, then this is equivalent to those modes if no mean decomp is used
                    #UNVERIFIED: That that property holds numerically with this implementation and whether it holds if using a mean decomp
                    if mean_reduction[j]!="zero":
                        print("WARNING: Correctness of approach unconfirmed for non-zero mean reduction")
                    uResults[i,:,2,:] = ((romData.uModes @ romData.uModes.transpose()) @ uResults[i,:,0,:])
                    vResults[i,:,2,:] = ((romData.vModes @ romData.vModes.transpose()) @ vResults[i,:,0,:])

        #================================================== Compute Error =================================================================
            for k in range(len(error_norm)):
                error[iret,j,k] = model.computeRomError(uResults[0,:,0,:],vResults[0,:,0,:],uResults[0,:,1,:],vResults[0,:,1,:],romData.W,norm=error_norm[k])
                print("ROM Error in norm "+error_norm[k]+": ", error[iret,j,k])
                #INCOMPLETE: Figure out what want to compute for sensitivity error.
        #=========================================== Make Movies ===================================================================
        if usePodRom and makeMovies:    
            subplotMovie(uResults, x, romSaveFolder + "u.mov", fps=15, xLabels="x", yLabels=uLabels, legends=legends, legendLoc="upper left", subplotSize=(2.5, 2))
            subplotMovie(vResults, x, romSaveFolder + "v.mov", fps=15, xLabels="x", yLabels=vLabels, legends=legends, legendLoc="upper left", subplotSize=(2.5, 2))
            subplotMovie(combinedResults, x, romSaveFolder + "combined.mov", fps=15, xLabels="x",  yLabels=combinedLabels, legends=legends, legendLoc="upper left", subplotSize=(2.5, 2))
        elif makeMovies:    
            subplotMovie(uResults, x, fomSaveFolder + "u.mov", fps=15, xLabels="x", yLabels=uLabels, legends=legends, legendLoc="upper left", subplotSize=(2.5, 2))
            subplotMovie(vResults, x, fomSaveFolder + "v.mov", fps=15, xLabels="x", yLabels=vLabels, legends=legends, legendLoc="upper left", subplotSize=(2.5, 2))
            subplotMovie(combinedResults, x, fomSaveFolder + "combined.mov", fps=15, xLabels="x",  yLabels=combinedLabels, legends=legends, legendLoc="upper left", subplotSize=(2.5, 2))

        #================================================== Make example plots ============================================================
        if plotTimeSeries:
                    tplot = np.linspace(0,tmax/tstep,4,dtype=int)
                    title = ["t=" + str(t*tstep) for t in tplot]
                    for i in range(len(uResults)):
                        uResults[i]=uResults[i][tplot]
                        vResults[i]=vResults[i][tplot]
                    for i in range(len(combinedResults)):
                        combinedResults[i]=combinedResults[i][tplot]

                    fig,axs = subplotTimeSeries(combinedResults, x, xLabels="x", yLabels=combinedLabels, title = title,legends=legends, subplotSize=(2.65, 2))
                    if usePodRom:
                        plt.savefig(romSaveFolder + "combinedTimeSeries.pdf", format="pdf")
                        plt.savefig(romSaveFolder + "combinedTimeSeries.png", format="png")
                    else:
                        plt.savefig(fomSaveFolder + "combinedTimeSeries.pdf", format="pdf")
                        plt.savefig(fomSaveFolder + "combinedTimeSeries.png", format="png")
                    fig,axs = subplotTimeSeries(uResults, x, xLabels="x", yLabels=uLabels, title = title,legends=legends, subplotSize=(2.65, 2))
                    if usePodRom:
                        plt.savefig(romSaveFolder + "uTimeSeries.pdf", format="pdf")
                        plt.savefig(romSaveFolder + "uTimeSeries.png", format="png")
                    else:
                        plt.savefig(fomSaveFolder + "uTimeSeries.pdf", format="pdf")
                        plt.savefig(fomSaveFolder + "uTimeSeries.png", format="png")
                    fig,axs = subplotTimeSeries(vResults, x, xLabels="x", yLabels=vLabels, title = title,legends=legends, subplotSize=(2.65, 2))
                    if usePodRom:
                        plt.savefig(romSaveFolder + "vTimeSeries.pdf", format="pdf")
                        plt.savefig(romSaveFolder + "vTimeSeries.png", format="png")
                    else:
                        plt.savefig(fomSaveFolder + "vTimeSeries.pdf", format="pdf")
                        plt.savefig(fomSaveFolder + "vTimeSeries.png", format="png")


#=========================================== Plot Error ===================================================================
if usePodRom and plotError and error.size>1:
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
    plt.savefig(fomSaveFolder + "/errorConvergence.pdf", format="pdf")
    plt.savefig(fomSaveFolder + "/errorConvergence.png", format="png")
plt.show()
