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
from tankModel.TankModel import TankModel
import matplotlib.pyplot as plt


#Set save details
paramSet = "BizonPeriodic" #BizonPeriodic, BizonLinear, BizonAdvecDiffusion
equationSet = "tankOnly" 
nCollocation=2
nElements=32
usePodRom=True
energyRetention=.99
xpoints=101
mean_reduction = "first"

#Notes on results


if paramSet == "BizonPeriodic":
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
    

saveLocation = "../../results/podRomAnalysis/"+paramSet +"_"+equationSet
saveLocation +="/nCol" + str(nCollocation) + "_nElem"+str(nElements)+"_e"+str(energyRetention)+"_"+mean_reduction+ "/"



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
if not os.path.exists(saveLocation):
        os.makedirs(saveLocation)
if usePodRom:
    saveLocation+="podRom_"
#==================================== Setup system ===============================================================================
model=TankModel(nCollocation=nCollocation,nElements=nElements,spacing="legendre",bounds=[0,1],params=baseParams)
dydtSens =lambda y,t: model.dydtSens(y,t,paramSelect=paramSelect)
if paramSet=="BizonAdvec":
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
    odeOut= scipy.integrate.solve_ivp(lambda t,y: dydtSens(y,t),(t,t+tstep),modelCoeff[-1], method='BDF',atol=1e-6,rtol=1e-6)
    change = np.max(np.abs(odeOut.y[:,-1]-modelCoeff[-1,:]))
    modelCoeff=np.append(modelCoeff,odeOut.y[:,[-1]].transpose(),axis=0)
    #print("(t,change): (", t,", ", change, ")")
    t+=tstep


#================================== Run POD-ROM ==================================================================================
if usePodRom:
    #------------------------------- Setup POD-ROM
    model.constructPodRom(modelCoeff[:,:2*nCollocation*nElements],x,energyRetention,mean=mean_reduction)
    nModesU = model.uModes.shape[1]
    nModesV = model.vModes.shape[1]
    print("Number of POD Modes for (u,v): (",nModesU, ", ",nModesV, ")")
    #Get Initial Modal Weights
    romCoeff=np.empty((1,neq*(nModesU+nModesV)))
    romCoeff[0,:nModesU]=model.uTimeModes[0,:]
    romCoeff[0,nModesU:nModesU+nModesV]=model.vTimeModes[0,:]
    for i in range(1, neq):
        start = i*(2*model.nCollocation*model.nElements)
        coeff = modelCoeff[0,start:start+2*model.nCollocation*model.nElements]
        uEval, vEval = model.eval(x,coeff,output="seperated")
        uModes = model.uModes @ uEval
        vModes = model.vModes @ vEval
        romCoeff[0,start:start+nModesU]=np.append(romCoeff,uModes,axis=0)
        romCoeff[0,start+nModesU:start+nModesU+nModesV]=np.append(romCoeff,vModes,axis=0)

    #------------------------------ Run POD-ROM
    dydtPodRom = lambda y,t: model.dydtPodRom(y,t,nModesU,nModesV,paramSelect=paramSelect,penaltyStrength=0)
    t=0
    while t<tmax:
        odeOut= scipy.integrate.solve_ivp(lambda t,y: dydtPodRom(y,t),(t,t+tstep),romCoeff[-1], method='BDF',atol=1e-6,rtol=1e-6)
        change = np.max(np.abs(odeOut.y[:,-1]-romCoeff[-1,:]))
        romCoeff=np.append(romCoeff,odeOut.y[:,[-1]].transpose(),axis=0)
        t+=tstep


#===========================================Make Movies ===================================================================

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
        romStart = i*(nModesU+nModesV)
        uFomResult, vFomResult = model.eval(x,modelCoeff[:,fomStart:fomStart+2*model.nCollocation*model.nElements],output="seperated")
        uRomResult = (model.uModes @ romCoeff[:,romStart:romStart+nModesU].transpose()).transpose() + model.uMean
        uResult[:,0,:]=uFomResult
        uResult[:,1,:]=uRomResult
        if i==0:
            uPOD = ((model.uModes @ model.uTimeModes.transpose())+model.uMean.reshape((model.uMean.size,1))).transpose()
            uResult[:,2,:]=uPOD
        uResults.append(uResult)
        combinedResults.append(uResult)


    for i in range(0, neq):
        if i==0:
            vResult = np.empty((modelCoeff.shape[0],3,x.size))
        else:
            vResult = np.empty((modelCoeff.shape[0],2,x.size))
        fomStart = i*(2*model.nCollocation*model.nElements)
        romStart = i*(nModesU+nModesV)
        uFomResult, vFomResult = model.eval(x,modelCoeff[:,fomStart:fomStart+2*model.nCollocation*model.nElements],output="seperated")
        vRomResult = (model.vModes @ romCoeff[:,romStart+nModesU:romStart+nModesU+nModesV].transpose()).transpose() + model.vMean
        vResult[:,0,:]=vFomResult
        vResult[:,1,:]=vRomResult
        if i==0:
            vPOD = (model.vModes@ model.vTimeModes.transpose() +model.vMean.reshape((model.vMean.size,1))).transpose()
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
    
subplotMovie(uResults, x, saveLocation + "u.mov", fps=15, xLabels="x", yLabels=uLabels, legends=legends, legendLoc="upper left", subplotSize=(2.5, 2))
subplotMovie(vResults, x, saveLocation + "v.mov", fps=15, xLabels="x", yLabels=vLabels, legends=legends, legendLoc="upper left", subplotSize=(2.5, 2))
subplotMovie(combinedResults, x, saveLocation + "combined.mov", fps=15, xLabels="x",  yLabels=combinedLabels, legends=legends, legendLoc="upper left", subplotSize=(2.5, 2))



#================================================== Make example plots ============================================================
tplot = np.linspace(0,tmax/tstep,5,dtype=int)
title = ["t=" + str(t*tstep) for t in tplot]
for i in range(len(uResults)):
    uResults[i]=uResults[i][tplot]
    vResults[i]=vResults[i][tplot]
for i in range(len(combinedResults)):
    combinedResults[i]=combinedResults[i][tplot]

fig,axs = subplotTimeSeries(combinedResults, x, xLabels="x", yLabels=combinedLabels, title = title,legends=legends, subplotSize=(2.5, 2))
plt.savefig(saveLocation + "combinedTimeSeries.pdf", format="pdf")
plt.savefig(saveLocation + "combinedTimeSeries.png", format="png")
fig,axs = subplotTimeSeries(uResults, x, xLabels="x", yLabels=uLabels, title = title,legends=legends, subplotSize=(2.5, 2))
plt.savefig(saveLocation + "uTimeSeries.pdf", format="pdf")
plt.savefig(saveLocation + "uTimeSeries.png", format="png")
fig,axs = subplotTimeSeries(vResults, x, xLabels="x", yLabels=vLabels, title = title,legends=legends, subplotSize=(2.5, 2))
plt.savefig(saveLocation + "vTimeSeries.pdf", format="pdf")
plt.savefig(saveLocation + "vTimeSeries.png", format="png")

plt.show()
