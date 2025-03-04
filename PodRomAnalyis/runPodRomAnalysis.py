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


#------------------------------- Setup test lists and error recording --------------------------
if type(modeRetention)==float or type(modeRetention)==int:
    modeRetention = [modeRetention]
if type(mean_reduction)==str:
    mean_reduction = [mean_reduction]
error = np.empty((len(modeRetention),len(mean_reduction),len(error_norm)))
truncationError=np.empty((len(modeRetention),len(mean_reduction),len(error_norm)))
for i in range(len(modeRetention)):
    for j in range(len(mean_reduction)):
#================================== Run POD-ROM ==================================================================================
        if usePodRom:
            #------------------------------- Make Folder to save data
            if plotTimeSeries or makeMovies:
                if useEnergyThreshold:
                    romSaveFolder = fomSaveFolder + "/podRom_e"+str(modeRetention[i])+"_"+mean_reduction[j]+ "/"
                else :
                    romSaveFolder = fomSaveFolder + "/podRom_n"+str(modeRetention[i])+"_"+mean_reduction[j]+ "/"

                if not os.path.exists(romSaveFolder):
                    os.makedirs(romSaveFolder)

            #------------------------------- Setup POD-ROM
            romData, truncationError[i,j,:]=model.constructPodRom(modelCoeff[:,:2*nCollocation*nElements],x,modeRetention[i],mean=mean_reduction[j],useEnergyThreshold=useEnergyThreshold)
            print("Number of POD Modes for (u,v): (",romData.uNmodes, ", ",romData.vNmodes, ")")
            #Get Initial Modal Weights
            romCoeff=np.empty((1,neq*(romData.uNmodes+romData.vNmodes)))
            romCoeff[0,:romData.uNmodes]=romData.uTimeModes[0,:]
            romCoeff[0,romData.uNmodes:romData.uNmodes+romData.vNmodes]=romData.vTimeModes[0,:]
            #Compute Time modes for sensitivity equations
            for i in range(1, neq):
                start = i*(2*model.nCollocation*model.nElements)
                coeff = modelCoeff[0,start:start+2*model.nCollocation*model.nElements]
                uEval, vEval = model.eval(x,coeff,output="seperated")
                uTimeModes = romData.uModes @ uEval
                vTimeModes = romData.vModes @ vEval
                romCoeff[0,start:start+romData.uNmodes]=np.append(romCoeff,uTimeModes,axis=0)
                romCoeff[0,start+romData.uNmodes:start+romData.uNmodes+romData.vNmodes]=np.append(romCoeff,vTimeModes,axis=0)

            #------------------------------ Run POD-ROM
            dydtPodRom = lambda y,t: model.dydtPodRom(y,t,romData,paramSelect=paramSelect,penaltyStrength=0)
            t=0
            while t<tmax:
                odeOut= scipy.integrate.solve_ivp(lambda t,y: dydtPodRom(y,t),(t,t+tstep),romCoeff[-1], method='BDF',atol=1e-10,rtol=1e-10)
                change = np.max(np.abs(odeOut.y[:,-1]-romCoeff[-1,:]))
                romCoeff=np.append(romCoeff,odeOut.y[:,[-1]].transpose(),axis=0)
                t+=tstep
                    #================================================== Compute Error =================================================================
            modelCoeff = modelCoeff[:romCoeff.shape[0],:]
            for k in range(len(error_norm)):
                print("ROM Error in norm "+error_norm[k]+": ", model.computeRomError(modelCoeff,romCoeff,romData,norm=error_norm[k]))
                error[i,j,k] = model.computeRomError(modelCoeff,romCoeff,romData,norm=error_norm[k])

        #===========================================Map Results Back into Spatial Space ===================================================================
        uResults = []
        vResults = []
        combinedResults=[]
        if usePodRom:
            legends=["FOM", "ROM", "POD"]
            for i in range(0, neq):
                if i==0:
                    uResult = np.empty((modelCoeff.shape[0],3,x.size))
                else:
                    uResult = np.empty((modelCoeff.shape[0],2,x.size))
                fomStart = i*(2*model.nCollocation*model.nElements)
                romStart = i*(romData.uNmodes+romData.vNmodes)
                uFomResult, vFomResult = model.eval(x,modelCoeff[:,fomStart:fomStart+2*model.nCollocation*model.nElements],output="seperated")
                uRomResult = (romData.uModes @ romCoeff[:,romStart:romStart+romData.uNmodes].transpose()).transpose() + romData.uMean
                uResult[:,0,:]=uFomResult
                uResult[:,1,:]=uRomResult
                if i==0:
                    uPOD = ((romData.uModes @ romData.uTimeModes[:romCoeff.shape[0],:].transpose())+romData.uMean.reshape((romData.uMean.size,1))).transpose()
                    uResult[:,2,:]=uPOD
                uResults.append(uResult)
                combinedResults.append(uResult)


            for i in range(0, neq):
                if i==0:
                    vResult = np.empty((modelCoeff.shape[0],3,x.size))
                else:
                    vResult = np.empty((modelCoeff.shape[0],2,x.size))
                fomStart = i*(2*model.nCollocation*model.nElements)
                romStart = i*(romData.uNmodes+romData.vNmodes)
                uFomResult, vFomResult = model.eval(x,modelCoeff[:,fomStart:fomStart+2*model.nCollocation*model.nElements],output="seperated")
                vRomResult = (romData.vModes @ romCoeff[:,romStart+romData.uNmodes:romStart+romData.uNmodes+romData.vNmodes].transpose()).transpose() + romData.vMean
                vResult[:,0,:]=vFomResult
                vResult[:,1,:]=vRomResult
                if i==0:
                    vPOD = (romData.vModes@ romData.vTimeModes[:romCoeff.shape[0],:].transpose() +romData.vMean.reshape((romData.vMean.size,1))).transpose()
                    vResult[:,2,:]=vPOD
                vResults.append(vResult)
                combinedResults.append(vResult)
        else:
            legends="null"
            for i in range(0, neq):
                fomStart = i*(2*model.nCollocation*model.nElements)
                uResult, vResult = model.eval(x,modelCoeff[:,fomStart:fomStart+2*model.nCollocation*model.nElements],output="seperated")
                uResults.append(uResult)
                combinedResults.append(uResult)

            for i in range(0, neq):
                fomStart = i*(2*model.nCollocation*model.nElements)
                uResult, vResult = model.eval(x,modelCoeff[:,fomStart:fomStart+2*model.nCollocation*model.nElements],output="seperated")
                vResults.append(vResult)
                combinedResults.append(vResult)
        
        
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
