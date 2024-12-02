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
nCollocation=2
nElements=32
resultsFolder = "../../results/podRomAnalysis/periodicExample_nCol" + str(nCollocation) + "_nElem"+str(nElements)+"/"
if not os.path.exists(resultsFolder):
        os.makedirs(resultsFolder)

usePodRom=True
#Determine parameters to get sensitivity of
neq=1
paramSelect=[]
uLabels=[r"$u$"]
vLabels=[r"$v$"]
combinedLabels= [r"$u$",r"$v$"]

# neq=5
# paramSelect=["Le","Da","delta","vH"]
# uLabels=[r"$u$",r"$u_{\mathrm{Le}}$",r"$u_{\mathrm{Da}}$",r"$u_{\delta}$",r"$u_{v_H}$"]
# vLabels=[r"$v$",r"$v_{\mathrm{Le}}$",r"$v_{\mathrm{Da}}$",r"$v_{\delta}$",r"$v_{v_H}$"]
# combinedLabels= [r"$u$",r"$u_{\mathrm{Le}}$",r"$u_{\mathrm{Da}}$",r"$u_{\delta}$",r"$u_{v_H}$",r"$v$",r"$v_{\mathrm{Le}}$",r"$v_{\mathrm{Da}}$",r"$v_{\delta}$",r"$v_{v_H}$"]

# neq=8
# paramSelect=["PeM","PeT","f","Le","Da","delta","vH"]
# uLabels=[r"$u$",r"$u_{\mathrm{Pe_M}}$",r"$u_{\mathrm{Pe_T}}$",r"$u_{f}$",r"$u_{\mathrm{Le}}$",r"$u_{\mathrm{Da}}$",r"$u_{\delta}$",r"$u_{v_H}$"]
# vLabels=[r"$v$",r"$v_{\mathrm{Pe_M}}$",r"$v_{\mathrm{Pe_T}}$",r"$v_{f}$",r"$v_{\mathrm{Le}}$",r"$v_{\mathrm{Da}}$",r"$v_{\delta}$",r"$v_{v_H}$"]
# combinedLabels= [r"$u$",r"$u_{\mathrm{Pe_M}}$",r"$u_{\mathrm{Pe_T}}$",r"$u_{f}$",r"$u_{\mathrm{Le}}$",r"$u_{\mathrm{Da}}$",r"$u_{\delta}$",r"$u_{v_H}$",r"$v$",r"$v_{\mathrm{Pe_M}}$",r"$v_{\mathrm{Pe_T}}$",r"$v_{f}$",r"$v_{\mathrm{Le}}$",r"$v_{\mathrm{Da}}$",r"$v_{\delta}$",r"$v_{v_H}$"]

# neq=10
# paramSelect=["PeM","PeT","f","Le","Da","beta","gamma","delta","vH"]
# uLabels=[r"$u$",r"$u_{\mathrm{Pe_M}}$",r"$u_{\mathrm{Pe_T}}$",r"$u_{f}$",r"$u_{\mathrm{Le}}$",r"$u_{\mathrm{Da}}$",r"$u_{\beta}$",r"$u_{\gamma}$",r"$u_{\delta}$",r"$u_{v_H}$"]
# vLabels=[r"$v$",r"$v_{\mathrm{Pe_M}}$",r"$v_{\mathrm{Pe_T}}$",r"$v_{f}$",r"$v_{\mathrm{Le}}$",r"$v_{\mathrm{Da}}$",r"$v_{\beta}$",r"$v_{\gamma}$",r"$v_{\delta}$",r"$v_{v_H}$"]
   

#==================================== Converge system in stable domain ===========================================================
# Use parameter test case in the periodic regime per Bizon analysis
# baseParams={"PeM": 300, "PeT": 300, "f": .3, "Le": 1, "Da": .15, "beta": 1.4, "gamma": 10,"delta": 2, "vH":-.02}
# model=TankModel(nCollocation=nCollocation,nElements=nElements,spacing="legendre",bounds=[0,1],params=baseParams)

# change=1
# t=0
# tstep=5

# while change > 1e-2 and t<200:
    
#     odeOut= scipy.integrate.solve_ivp(lambda t,y: dydtSens(y,t),(t,t+tstep),modelCoeff[-1], method='BDF',atol=1e-6,rtol=1e-6)
#     change = np.max(np.abs(odeOut.y[:,-1]-modelCoeff[-1,:]))
#     modelCoeff=np.append(modelCoeff,odeOut.y[:,[-1]].transpose(),axis=0)
#     print("(t,change): (", t,", ", change, ")")
#     t+=tstep

#==================================== Setup system for periodic domain ===========================================================
baseParams={"PeM": 300, "PeT": 300, "f": .3, "Le": 1, "Da": .15, "beta": 1.4, "gamma": 10,"delta": 2, "vH":-.045}
model=TankModel(nCollocation=nCollocation,nElements=nElements,spacing="legendre",bounds=[0,1],params=baseParams)
modelCoeff=np.ones((1,model.nCollocation*model.nElements*2*neq))
x=np.linspace(0,1,51)
#Run out till stabalizing in periodic domain
dydtSens =lambda y,t: model.dydtSens(y,t,paramSelect=paramSelect)
odeOut= scipy.integrate.solve_ivp(lambda t,y: dydtSens(y,t),(0,100),modelCoeff[-1,:], method='BDF',atol=1e-6,rtol=1e-6)
modelCoeff = odeOut.y[:,[-1]].transpose()
#=================================== Run system in periodic domain ================================================================

t=0
tstep=.05
tmax=3
while t<tmax:
    odeOut= scipy.integrate.solve_ivp(lambda t,y: dydtSens(y,t),(t,t+tstep),modelCoeff[-1], method='BDF',atol=1e-6,rtol=1e-6)
    change = np.max(np.abs(odeOut.y[:,-1]-modelCoeff[-1,:]))
    modelCoeff=np.append(modelCoeff,odeOut.y[:,[-1]].transpose(),axis=0)
    #print("(t,change): (", t,", ", change, ")")
    t+=tstep


#================================== Run POD-ROM ==================================================================================
if usePodRom:
    #------------------------------- Setup POD-ROM
    model.constructPodRom(modelCoeff[:,:2*nCollocation*nElements],x,.99)
    nModesU = model.uModes.shape[1]
    nModesV = model.vModes.shape[1]
    print("Number of POD Modes for (u,v): (",nModesU, ", ",nModesV, ")")
    #Get Initial Modal Weights
    romCoeff=np.empty((1,neq*(nModesU+nModesV)))
    romCoeff[0,:nModesU]=model.uTimeModes[:,0]
    romCoeff[0,nModesU:nModesU+nModesV]=model.vTimeModes[:,0]
    #------------------------------ Run POD-ROM
    for i in range(1, neq):
        start = i*(2*model.nCollocation*model.nElements)
        coeff = modelCoeff[0,start:start+2*model.nCollocation*model.nElements]
        uEval, vEval = model.eval(x,coeff,output="seperated")
        uModes = model.uModes @ uEval
        vModes = model.vModes @ vEval
        romCoeff[0,start:start+nModesU]=np.append(romCoeff,uModes,axis=0)
        romCoeff[0,start+nModesU:start+nModesU+nModesV]=np.append(romCoeff,vModes,axis=0)

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
            uPOD = ((model.uModes @ model.uTimeModes)+model.uMean.reshape((model.uMean.size,1))).transpose()
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
            vPOD = (model.vModes@ model.vTimeModes +model.vMean.reshape((model.vMean.size,1))).transpose()
            vResult[:,2,:]=vPOD
        vResults.append(vResult)
        combinedResults.append(vResult)
else:
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
    
subplotMovie(uResults, x, resultsFolder + "uPeriodic.mov", fps=15, xLabels="x", yLabels=uLabels, legends=["FOM", "ROM", "POD"], legendLoc="upper left", subplotSize=(2.5, 2))
subplotMovie(vResults, x, resultsFolder + "vPeriodic.mov", fps=15, xLabels="x", yLabels=vLabels, legends=["FOM", "ROM","POD"], legendLoc="upper left", subplotSize=(2.5, 2))
subplotMovie(combinedResults, x, resultsFolder + "combinedPeriodic.mov", fps=15, xLabels="x", legends=["FOM", "ROM","POD"], legendLoc="upper left", yLabels=combinedLabels, subplotSize=(2.5, 2))

tplot = np.linspace(0,tmax/tstep,5,dtype=int)
title = ["t=" + str(t*tstep) for t in tplot]
uResults = []
vResults = []
for i in range(0, neq):
    start = i*(2*model.nCollocation*model.nElements)
    coeff = modelCoeff[tplot,start:start+2*model.nCollocation*model.nElements]
    uResult, vResult = model.eval(x,coeff,output="seperated")
    uResults.append(uResult)
    vResults.append(vResult)
fig,axs = subplotTimeSeries(uResults, x, xLabels="x", yLabels=uLabels, title = title, subplotSize=(2.5, 2))
plt.savefig(resultsFolder + "uTimeSeries.pdf", format="pdf")
plt.savefig(resultsFolder + "uTimeSeries.png", format="png")
fig,axs = subplotTimeSeries(vResults, x, xLabels="x", yLabels=vLabels, title = title, subplotSize=(2.5, 2))
plt.savefig(resultsFolder + "vTimeSeries.pdf", format="pdf")
plt.savefig(resultsFolder + "vTimeSeries.png", format="png")

plt.show()
