import sys
import os 
current_script_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath(os.path.join(current_script_dir,'..','..'))
sys.path.append(grandparent_dir)
print(sys.path)

import numpy as np
from tankModel.TankModel import TankModel
import scipy




print("Testing Sensitivity Equations")
print("     Testing vH")
#========================================== vH Tests ==================================================================
#Case 1: Transport and Diffusion Portion
#Set Da=0 so the nonlinear source term is 0
params={"PeM": 300, "PeT": 100, "f": 0, "Le": 3, "Da": 0, "beta": 0, "gamma": 0,"delta": 2, "vH":1}
#Define nElem=1,nColl=2 model with roots at -1 and 1 for simplicity
model = TankModel(nCollocation=1,nElements=2,spacing="legendre",bounds=[0,1],params=params)
x=model.collocationPoints
#Case 1a: u=x^2, uvH=x^2, v=x^2, vvH=x^2
u = -x**2+2*x+2/params["PeM"]
v = -x**2+2*x+(2/params["PeT"]+params["f"]*(2-1))/(1-params["f"])
dudvH = -x**2+2*x+2/params["PeM"]
dvdvH = -x**2+2*x+(2/params["PeT"]+params["f"]*(2-1))/(1-params["f"])
dudvHx = -2*x+2
dvdvHx = -2*x+2
dudvHxx = -2
dvdvHxx = -2
y=np.concatenate((u,v,dudvH,dvdvH),axis=0)
dydt = model.dydtSens(y,0,paramSelect=["vH"])
dudvHComputed = dydt[2*model.nCollocation*model.nElements:3*model.nCollocation*model.nElements]
dvdvHComputed = dydt[3*model.nCollocation*model.nElements:]
#Check ddudvHdt
assert(np.isclose(dudvHComputed, -dudvHx+dudvHxx/params["PeM"]                                        ).all())
#Check ddvdvHdt
assert(np.isclose(dvdvHComputed,(-dvdvHx+dvdvHxx/params["PeT"]+params["delta"]*(1-dvdvH))/params["Le"]).all())

#Case 2: Nonlinear Source Portion
params={"PeM": 200, "PeT": 100, "f": .4, "Le": 2, "Da": .8, "beta": 10, "gamma": 1.1,"delta": 2, "vH": 4}
#Define nElem=1,nColl=2 model with roots at -1 and 1 for simplicity
model = TankModel(nCollocation=1,nElements=2,spacing="legendre",bounds=[0,1],params=params)
x=model.collocationPoints
#Case 1a: u=x^2, uvH=x^2, v=x^2, vvH=x^2
u = -x**2+2*x+2/params["PeM"]
v = -x**2+2*x+(2/params["PeT"]+params["f"]*(2-1))/(1-params["f"])
dudvH = -x**2+2*x+2/params["PeM"]
dvdvH = -x**2+2*x+(2/params["PeT"]+params["f"]*(2-1))/(1-params["f"])
dudvHx = -2*x+2
dvdvHx = -2*x+2
dudvHxx = -2
dvdvHxx = -2
y=np.concatenate((u,v,dudvH,dvdvH),axis=0)
dydt = model.dydtSens(y,0,paramSelect=["vH"])
dudvHComputed = dydt[2*model.nCollocation*model.nElements:3*model.nCollocation*model.nElements]
dvdvHComputed = dydt[3*model.nCollocation*model.nElements:]
#Check ddudvHdt
assert(np.isclose(dudvHComputed, -dudvHx+dudvHxx/params["PeM"]\
                                    +params["Da"]*np.exp(params["gamma"]*params["beta"]*v/(1+params["beta"]*v))\
                                    *((1-u)*(params["gamma"]*params["beta"]*dvdvH/(1+params["beta"]*v)**2)-dudvH)).all())
#Check ddvdvHdt
assert(np.isclose(dvdvHComputed,(-dvdvHx+dvdvHxx/params["PeT"]\
                                  +params["Da"]*np.exp(params["gamma"]*params["beta"]*v/(1+params["beta"]*v))\
                                    *((1-u)*(params["gamma"]*params["beta"]*dvdvH/(1+params["beta"]*v)**2)-dudvH)\
                                    +params["delta"]*(1-dvdvH))/params["Le"]).all())
print("     vH Sensitivity Passes")

#========================================== delta Tests ==================================================================
print("     Testing delta")
#Case 1: Transport and Diffusion Portion
#Set Da=0 so the nonlinear source term is 0
params={"PeM": 300, "PeT": 100, "f": 0, "Le": 3, "Da": 0, "beta": 0, "gamma": 0,"delta": 2, "vH":1}
#Define nElem=1,nColl=2 model with roots at -1 and 1 for simplicity
model = TankModel(nCollocation=3,nElements=2,spacing="legendre",bounds=[0,1],params=params)
x=model.collocationPoints
#Case 1a: u=x^2, uvH=x^2, v=x^2, vvH=x^2
u = -x**2+2*x+2/params["PeM"]
v = -x**2+2*x+(2/params["PeT"]+params["f"]*(2-1))/(1-params["f"])
dudDelta = -x**3-x**2+5*x+5/params["PeM"]
dvdDelta = -x**3-x**2+5*x+(5/params["PeT"]+params["f"]*(5-2))/(1-params["f"])
dudDeltax = -3*x**2-2*x+5
dvdDeltax = -3*x**2-2*x+5
dudDeltaxx = -6*x-2
dvdDeltaxx = -6*x-2
y=np.concatenate((u,v,dudDelta,dvdDelta),axis=0)
dydt = model.dydtSens(y,0,paramSelect=["delta"])
dudDeltaComputed = dydt[2*model.nCollocation*model.nElements:3*model.nCollocation*model.nElements]
dvdDeltaComputed = dydt[3*model.nCollocation*model.nElements:]
#Check ddudDeltadt
assert(np.isclose(dudDeltaComputed, -dudDeltax+dudDeltaxx/params["PeM"]                                        ).all())
#Check ddvdDeltadt
assert(np.isclose(dvdDeltaComputed,(-dvdDeltax+dvdDeltaxx/params["PeT"]+params["vH"]-v-params["delta"]*dvdDelta)/params["Le"]).all())

#Case 2: Nonlinear Source Portion
params={"PeM": 200, "PeT": 100, "f": .4, "Le": 2, "Da": .8, "beta": 10, "gamma": 1.1,"delta": 2, "vH": 4}
#Define nElem=1,nColl=2 model with roots at -1 and 1 for simplicity
model = TankModel(nCollocation=1,nElements=2,spacing="legendre",bounds=[0,1],params=params)
x=model.collocationPoints
#Case 1a: u=x^2, uDelta=x^2, v=x^2, vDelta=x^2
u = -x**3-x**2+5*x+5/params["PeM"]
v = -x**3-x**2+5*x+(3/params["PeT"]+params["f"]*(5-2))/(1-params["f"])
dudDelta = -x**2+2*x+2/params["PeM"]
dvdDelta = -x**2+2*x+(2/params["PeT"]+params["f"]*(2-1))/(1-params["f"])
dudDeltax = -2*x+2
dvdDeltax = -2*x+2
dudDeltaxx = -2
dvdDeltaxx = -2
y=np.concatenate((u,v,dudDelta,dvdDelta),axis=0)
dydt = model.dydtSens(y,0,paramSelect=["delta"])
dudDeltaComputed = dydt[2*model.nCollocation*model.nElements:3*model.nCollocation*model.nElements]
dvdDeltaComputed = dydt[3*model.nCollocation*model.nElements:]
#Check ddudDeltadt
assert(np.isclose(dudDeltaComputed, -dudDeltax+dudDeltaxx/params["PeM"]\
                                    +params["Da"]*np.exp(params["gamma"]*params["beta"]*v/(1+params["beta"]*v))\
                                    *((1-u)*(params["gamma"]*params["beta"]*dvdDelta/(1+params["beta"]*v)**2)-dudDelta)).all())
#Check ddvdDeltadt
assert(np.isclose(dvdDeltaComputed,(-dvdDeltax+dvdDeltaxx/params["PeT"]\
                                  +params["Da"]*np.exp(params["gamma"]*params["beta"]*v/(1+params["beta"]*v))\
                                    *((1-u)*(params["gamma"]*params["beta"]*dvdDelta/(1+params["beta"]*v)**2)-dudDelta)\
                                    +params["vH"]-v-params["delta"]*dvdDelta)/params["Le"]).all())
print("     delta Sensitivity Passes")


#========================================== Da Tests ==================================================================
print("     Testing Da")
#Skip Case 1 without nonlinear term since it can be removed since not all terms are multiplied by Da
#Case 2: Nonlinear Source Portion
params={"PeM": 200, "PeT": 100, "f": .4, "Le": 2, "Da": .8, "beta": 10, "gamma": 1.1,"delta": 2, "vH": 4}
#Define nElem=1,nColl=2 model with roots at -1 and 1 for simplicity
model = TankModel(nCollocation=3,nElements=2,spacing="legendre",bounds=[0,1],params=params)
x=model.collocationPoints
#Case 1a: u=x^2, uDa=x^2, v=x^2, vDa=x^2
u = -x**3-x**2+5*x+5/params["PeM"]
v = -x**3-x**2+5*x+(3/params["PeT"]+params["f"]*(5-2))/(1-params["f"])
dudDa = -x**3-x**2+5*x+5/params["PeM"]
dvdDa = -x**3-x**2+5*x+(5/params["PeT"]+params["f"]*(5-2))/(1-params["f"])
dudDax = -3*x**2-2*x+5
dvdDax = -3*x**2-2*x+5
dudDaxx = -6*x-2
dvdDaxx = -6*x-2
y=np.concatenate((u,v,dudDa,dvdDa),axis=0)
dydt = model.dydtSens(y,0,paramSelect=["Da"])
dudDaComputed = dydt[2*model.nCollocation*model.nElements:3*model.nCollocation*model.nElements]
dvdDaComputed = dydt[3*model.nCollocation*model.nElements:]
#Check ddudDadt
assert(np.isclose(dudDaComputed, -dudDax+dudDaxx/params["PeM"]\
                                    +np.exp(params["gamma"]*params["beta"]*v/(1+params["beta"]*v))*(1-u+params["Da"]*(\
                                      (1-u)*(params["gamma"]*params["beta"]*dvdDa/(1+params["beta"]*v)**2)-dudDa))).all())
#Check ddvdDadt
assert(np.isclose(dvdDaComputed,(-dvdDax+dvdDaxx/params["PeT"]-params["delta"]*dvdDa\
                                  +np.exp(params["gamma"]*params["beta"]*v/(1+params["beta"]*v))*(1-u+params["Da"]*(\
                                      (1-u)*(params["gamma"]*params["beta"]*dvdDa/(1+params["beta"]*v)**2)-dudDa)))/params["Le"]).all())
print("     Da Sensitivity Passes")


#========================================== gamma Tests ==================================================================
print("     Testing gamma")
#Case 1: Transport and Diffusion Portion
#Set Da=0 so the nonlinear source term is 0
params={"PeM": 300, "PeT": 100, "f": 0, "Le": 3, "Da": 0, "beta": 0, "gamma": 0,"delta": 2, "vH":1}
#Define nElem=1,nColl=2 model with roots at -1 and 1 for simplicity
model = TankModel(nCollocation=3,nElements=2,spacing="legendre",bounds=[0,1],params=params)
x=model.collocationPoints
#Case 1a: u=x^2, uvH=x^2, v=x^2, vvH=x^2
u = -x**2+2*x+2/params["PeM"]
v = -x**2+2*x+(2/params["PeT"]+params["f"]*(2-1))/(1-params["f"])
dudgamma = -x**3-x**2+5*x+5/params["PeM"]
dvdgamma = -x**3-x**2+5*x+(5/params["PeT"]+params["f"]*(5-2))/(1-params["f"])
dudgammax = -3*x**2-2*x+5
dvdgammax = -3*x**2-2*x+5
dudgammaxx = -6*x-2
dvdgammaxx = -6*x-2
y=np.concatenate((u,v,dudgamma,dvdgamma),axis=0)
dydt = model.dydtSens(y,0,paramSelect=["gamma"])
dudgammaComputed = dydt[2*model.nCollocation*model.nElements:3*model.nCollocation*model.nElements]
dvdgammaComputed = dydt[3*model.nCollocation*model.nElements:]
#Check ddudgammadt
assert(np.isclose(dudgammaComputed, -dudgammax+dudgammaxx/params["PeM"]                                        ).all())
#Check ddvdgammadt
assert(np.isclose(dvdgammaComputed,(-dvdgammax+dvdgammaxx/params["PeT"]-params["delta"]*dvdgamma)/params["Le"]).all())

#Case 2: Nonlinear Source Portion
params={"PeM": 200, "PeT": 100, "f": .4, "Le": 2, "Da": .8, "beta": 10, "gamma": 1.1,"delta": 2, "vH": 4}
#Define nElem=1,nColl=2 model with roots at -1 and 1 for simplicity
model = TankModel(nCollocation=4,nElements=2,spacing="legendre",bounds=[0,1],params=params)
x=model.collocationPoints
#Case 1a: u=x^2, ugamma=x^2, v=x^2, vgamma=x^2
u = -x**3-x**2+5*x+5/params["PeM"]
v = -x**3-x**2+5*x+(3/params["PeT"]+params["f"]*(5-2))/(1-params["f"])
dudgamma = -x**2+2*x+2/params["PeM"]
dvdgamma = -x**2+2*x+(2/params["PeT"]+params["f"]*(2-1))/(1-params["f"])
dudgammax = -2*x+2
dvdgammax = -2*x+2
dudgammaxx = -2
dvdgammaxx = -2
y=np.concatenate((u,v,dudgamma,dvdgamma),axis=0)
dydt = model.dydtSens(y,0,paramSelect=["gamma"])
dudgammaComputed = dydt[2*model.nCollocation*model.nElements:3*model.nCollocation*model.nElements]
dvdgammaComputed = dydt[3*model.nCollocation*model.nElements:]
#Check ddudgammadt
assert(np.isclose(dudgammaComputed, -dudgammax+dudgammaxx/params["PeM"]\
                                    +params["Da"]*np.exp(params["gamma"]*params["beta"]*v/(1+params["beta"]*v))\
                                    *((1-u)*params["beta"]*(v+params["beta"]*v**2+params["gamma"]*dvdgamma)/(1+params["beta"]*v)**2\
                                    -dudgamma)).all())
#Check ddvdgammadt
assert(np.isclose(dvdgammaComputed,(-dvdgammax+dvdgammaxx/params["PeT"]-params["delta"]*dvdgamma\
                                  +params["Da"]*np.exp(params["gamma"]*params["beta"]*v/(1+params["beta"]*v))\
                                    *((1-u)*params["beta"]*(v+params["beta"]*v**2+params["gamma"]*dvdgamma)/(1+params["beta"]*v)**2\
                                    -dudgamma))/params["Le"]).all())
print("     gamma Sensitivity Passes")

#========================================== beta Tests ==================================================================
print("     Testing beta")
#Case 1: Transport and Diffusion Portion
#Set Da=0 so the nonlinear source term is 0
params={"PeM": 300, "PeT": 100, "f": 0, "Le": 3, "Da": 0, "beta": 0, "gamma": 0,"delta": 2, "vH":1}
#Define nElem=1,nColl=2 model with roots at -1 and 1 for simplicity
model = TankModel(nCollocation=3,nElements=2,spacing="legendre",bounds=[0,1],params=params)
x=model.collocationPoints
#Case 1a: u=x^2, uvH=x^2, v=x^2, vvH=x^2
u = -x**2+2*x+2/params["PeM"]
v = -x**2+2*x+(2/params["PeT"]+params["f"]*(2-1))/(1-params["f"])
dudbeta = -x**3-x**2+5*x+5/params["PeM"]
dvdbeta = -x**3-x**2+5*x+(5/params["PeT"]+params["f"]*(5-2))/(1-params["f"])
dudbetax = -3*x**2-2*x+5
dvdbetax = -3*x**2-2*x+5
dudbetaxx = -6*x-2
dvdbetaxx = -6*x-2
y=np.concatenate((u,v,dudbeta,dvdbeta),axis=0)
dydt = model.dydtSens(y,0,paramSelect=["beta"])
dudbetaComputed = dydt[2*model.nCollocation*model.nElements:3*model.nCollocation*model.nElements]
dvdbetaComputed = dydt[3*model.nCollocation*model.nElements:]
#Check ddudbetadt
assert(np.isclose(dudbetaComputed, -dudbetax+dudbetaxx/params["PeM"]                                        ).all())
#Check ddvdbetadt
assert(np.isclose(dvdbetaComputed,(-dvdbetax+dvdbetaxx/params["PeT"]-params["delta"]*dvdbeta)/params["Le"]).all())

#Case 2: Nonlinear Source Portion
params={"PeM": 200, "PeT": 100, "f": .4, "Le": 2, "Da": .8, "beta": 10, "gamma": 1.1,"delta": 2, "vH": 4}
#Define nElem=1,nColl=2 model with roots at -1 and 1 for simplicity
model = TankModel(nCollocation=1,nElements=2,spacing="legendre",bounds=[0,1],params=params)
x=model.collocationPoints
#Case 1a: u=x^2, ubeta=x^2, v=x^2, vbeta=x^2
u = -x**3-x**2+5*x+5/params["PeM"]
v = -x**3-x**2+5*x+(3/params["PeT"]+params["f"]*(5-2))/(1-params["f"])
dudbeta = -x**2+2*x+2/params["PeM"]
dvdbeta = -x**2+2*x+(2/params["PeT"]+params["f"]*(2-1))/(1-params["f"])
dudbetax = -2*x+2
dvdbetax = -2*x+2
dudbetaxx = -2
dvdbetaxx = -2
y=np.concatenate((u,v,dudbeta,dvdbeta),axis=0)
dydt = model.dydtSens(y,0,paramSelect=["beta"])
dudbetaComputed = dydt[2*model.nCollocation*model.nElements:3*model.nCollocation*model.nElements]
dvdbetaComputed = dydt[3*model.nCollocation*model.nElements:]
#Check ddudbetadt
assert(np.isclose(dudbetaComputed, -dudbetax+dudbetaxx/params["PeM"]\
                                    +params["Da"]*np.exp(params["gamma"]*params["beta"]*v/(1+params["beta"]*v))\
                                    *((1-u)*params["gamma"]*(v+params["beta"]*dvdbeta)/(1+params["beta"]*v)**2\
                                    -dudbeta)).all())
#Check ddvdgammadt
assert(np.isclose(dvdbetaComputed,(-dvdbetax+dvdbetaxx/params["PeT"]-params["delta"]*dvdbeta\
                                  +params["Da"]*np.exp(params["gamma"]*params["beta"]*v/(1+params["beta"]*v))\
                                    *((1-u)*params["gamma"]*(v+params["beta"]*dvdbeta)/(1+params["beta"]*v)**2\
                                    -dudbeta))/params["Le"]).all())
print("     beta Sensitivity Passes")

#========================================== Le Tests ==================================================================
print("     Testing Le")
#Case 1: Transport and Diffusion Portion
#Set Da=0 so the nonlinear source term is 0
params={"PeM": 300, "PeT": 100, "f": 0, "Le": 3, "Da": 0, "beta": 0, "gamma": 0,"delta": 2, "vH":1}
#Define nElem=1,nColl=2 model with roots at -1 and 1 for simplicity
model = TankModel(nCollocation=3,nElements=2,spacing="legendre",bounds=[0,1],params=params)
x=model.collocationPoints
#Case 1a: u=x^2, uvH=x^2, v=x^2, vvH=x^2
u = -x**2+2*x+2/params["PeM"]
v = -x**2+2*x+(2/params["PeT"]+params["f"]*(2-1))/(1-params["f"])
dudLe = -x**3-x**2+5*x+5/params["PeM"]
dvdLe = -x**3-x**2+5*x+(5/params["PeT"]+params["f"]*(5-2))/(1-params["f"])
dudLex = -3*x**2-2*x+5
dvdLex = -3*x**2-2*x+5
dudLexx = -6*x-2
dvdLexx = -6*x-2
y=np.concatenate((u,v,dudLe,dvdLe),axis=0)
dydt = model.dydtSens(y,0,paramSelect=["Le"])
dvdt = dydt[model.nCollocation*model.nElements:2*model.nCollocation*model.nElements]
dudLeComputed = dydt[2*model.nCollocation*model.nElements:3*model.nCollocation*model.nElements]
dvdLeComputed = dydt[3*model.nCollocation*model.nElements:]
#Check ddudLedt
assert(np.isclose(dudLeComputed, -dudLex+dudLexx/params["PeM"]).all())
#Check ddvdLedt
assert(np.isclose(dvdLeComputed,(-dvdLex-dvdt+dvdLexx/params["PeT"]-params["delta"]*dvdLe)/params["Le"]).all())

#Case 2: Nonlinear Source Portion
params={"PeM": 200, "PeT": 100, "f": .4, "Le": 2, "Da": .8, "beta": 10, "gamma": 1.1,"delta": 2, "vH": 4}
#Define nElem=1,nColl=2 model with roots at -1 and 1 for simplicity
model = TankModel(nCollocation=1,nElements=2,spacing="legendre",bounds=[0,1],params=params)
x=model.collocationPoints
#Case 1a: u=x^2, uLe=x^2, v=x^2, vLe=x^2
u = -x**3-x**2+5*x+5/params["PeM"]
v = -x**3-x**2+5*x+(3/params["PeT"]+params["f"]*(5-2))/(1-params["f"])
dudLe = -x**2+2*x+2/params["PeM"]
dvdLe = -x**2+2*x+(2/params["PeT"]+params["f"]*(2-1))/(1-params["f"])
dudLex = -2*x+2
dvdLex = -2*x+2
dudLexx = -2
dvdLexx = -2
y=np.concatenate((u,v,dudLe,dvdLe),axis=0)
dydt = model.dydtSens(y,0,paramSelect=["Le"])
dvdt = dydt[model.nCollocation*model.nElements:2*model.nCollocation*model.nElements]
dudLeComputed = dydt[2*model.nCollocation*model.nElements:3*model.nCollocation*model.nElements]
dvdLeComputed = dydt[3*model.nCollocation*model.nElements:]
#Check ddudLedt
#Check ddudvHdt
assert(np.isclose(dudLeComputed, -dudLex+dudLexx/params["PeM"]\
                                    +params["Da"]*np.exp(params["gamma"]*params["beta"]*v/(1+params["beta"]*v))\
                                    *((1-u)*(params["gamma"]*params["beta"]*dvdLe/(1+params["beta"]*v)**2)-dudLe)).all())
#Check ddvdLedt
assert(np.isclose(dvdLeComputed,(-dvdt-dvdLex+dvdLexx/params["PeT"]-params["delta"]*dvdLe\
                                  +params["Da"]*np.exp(params["gamma"]*params["beta"]*v/(1+params["beta"]*v))\
                                    *((1-u)*(params["gamma"]*params["beta"]*dvdLe/(1+params["beta"]*v)**2)-dudLe))/params["Le"]).all())
print("     Le Sensitivity Passes")

#========================================== PeM Tests ==================================================================
print("     Testing PeM")
#Case 1: Transport and Diffusion Portion
#Set Da=0 so the nonlinear source term is 0
params={"PeM": 1000, "PeT": 100, "f": 0, "Le": 3, "Da": 0, "beta": 0, "gamma": 0,"delta": 2, "vH":1}
#Define nElem=1,nColl=2 model with roots at -1 and 1 for simplicity
model = TankModel(nCollocation=2,nElements=2,spacing="legendre",bounds=[0,1],params=params)
x=model.collocationPoints
#Case 1a: u=x^2, uvH=x^2, v=x^2, vvH=x^2
u = -x**3-x**2+5*x+5/params["PeM"]
uxx= -6*x-2
v = -x**2+2*x+(2/params["PeT"]+params["f"]*(2-1))/(1-params["f"])
dudPeM = -x**3-x**2+5*x+(5-5/params["PeM"])/params["PeM"]
dudPeMx = -3*x**2-2*x+5
dudPeMxx = -6*x-2
dvdPeM = -x**3-x**2+5*x+(5/params["PeT"]+params["f"]*(5-2))/(1-params["f"])
dvdPeMx = -3*x**2-2*x+5
dvdPeMxx = -6*x-2
y=np.concatenate((u,v,dudPeM,dvdPeM),axis=0)
dydt = model.dydtSens(y,0,paramSelect=["PeM"])
dvdt = dydt[model.nCollocation*model.nElements:2*model.nCollocation*model.nElements]
dudPeMComputed = dydt[2*model.nCollocation*model.nElements:3*model.nCollocation*model.nElements]
dvdPeMComputed = dydt[3*model.nCollocation*model.nElements:]
#Check ddudPeMdt
assert(np.isclose(dudPeMComputed, -dudPeMx+dudPeMxx/params["PeM"]-uxx/(params["PeM"]**2)).all())
#Check ddvdPeMdt
assert(np.isclose(dvdPeMComputed,(-dvdPeMx+dvdPeMxx/params["PeT"]-params["delta"]*dvdPeM)/params["Le"]).all())

#Case 2: Nonlinear Source Portion
params={"PeM": 200, "PeT": 100, "f": .4, "Le": 2, "Da": .8, "beta": 10, "gamma": 1.1,"delta": 2, "vH": 4}
#Define nElem=1,nColl=2 model with roots at -1 and 1 for simplicity
model = TankModel(nCollocation=4,nElements=3,spacing="legendre",bounds=[0,1],params=params)
x=model.collocationPoints
#Case 1a: u=x^2, uPeM=x^2, v=x^2, vPeM=x^2
u = -x**3-x**2+5*x+5/params["PeM"]
uxx= -6*x-2
v = -x**2+2*x+(2/params["PeT"]+params["f"]*(2-1))/(1-params["f"])
dudPeM = -x**3-x**2+5*x+(5-5/params["PeM"])/params["PeM"]
dudPeMx = -3*x**2-2*x+5
dudPeMxx = -6*x-2
dvdPeM = -x**3-x**2+5*x+(5/params["PeT"]+params["f"]*(5-2))/(1-params["f"])
dvdPeMx = -3*x**2-2*x+5
dvdPeMxx = -6*x-2
y=np.concatenate((u,v,dudPeM,dvdPeM),axis=0)
dydt = model.dydtSens(y,0,paramSelect=["PeM"])
dudPeMComputed = dydt[2*model.nCollocation*model.nElements:3*model.nCollocation*model.nElements]
dvdPeMComputed = dydt[3*model.nCollocation*model.nElements:]
#Check ddudPeMdt
assert(np.isclose(dudPeMComputed, -dudPeMx+dudPeMxx/params["PeM"]-uxx/(params["PeM"]**2)\
                                    +params["Da"]*np.exp(params["gamma"]*params["beta"]*v/(1+params["beta"]*v))\
                                    *((1-u)*(params["gamma"]*params["beta"]*dvdPeM/(1+params["beta"]*v)**2)-dudPeM)).all())
#Check ddvdPeMdt
assert(np.isclose(dvdPeMComputed,(-dvdPeMx+dvdPeMxx/params["PeT"]-params["delta"]*dvdPeM\
                                  +params["Da"]*np.exp(params["gamma"]*params["beta"]*v/(1+params["beta"]*v))\
                                    *((1-u)*(params["gamma"]*params["beta"]*dvdPeM/(1+params["beta"]*v)**2)-dudPeM))/params["Le"]).all())
print("     PeM Sensitivity Passes")

#========================================== PeT Tests ==================================================================
print("     Testing PeT")
#Case 1: Transport and Diffusion Portion
#Set Da=0 so the nonlinear source term is 0
params={"PeM": 30, "PeT": 20, "f": 0, "Le": 3, "Da": 0, "beta": 0, "gamma": 0,"delta": 2, "vH":1}
#Define nElem=1,nColl=2 model with roots at -1 and 1 for simplicity
model = TankModel(nCollocation=2,nElements=2,spacing="legendre",bounds=[0,1],params=params)
x=model.collocationPoints
#Case 1a: u=x^2, uvH=x^2, v=x^2, vvH=x^2
u = -x**3-x**2+5*x+5/params["PeM"]
v = lambda x:-x**2+2*x+(2/params["PeT"]+params["f"]*(2-1))/(1-params["f"])
vxx=-2 + x*0
dudPeT = -x**3-x**2+5*x+5/params["PeM"]
dudPeTx = -3*x**2-2*x+5
dudPeTxx = -6*x-2
dvdPeT = -x**3-x**2+5*x+((5+params["f"]*v(1)-v(0))/params["PeT"]+params["f"]*(5-2))/(1-params["f"])
dvdPeTx = -3*x**2-2*x+5
dvdPeTxx = -6*x-2
y=np.concatenate((u,v(x),dudPeT,dvdPeT),axis=0)
dydt = model.dydtSens(y,0,paramSelect=["PeT"])
dudPeTComputed = dydt[2*model.nCollocation*model.nElements:3*model.nCollocation*model.nElements]
dvdPeTComputed = dydt[3*model.nCollocation*model.nElements:]
#Check ddudPeMdt
assert(np.isclose(dudPeTComputed, -dudPeTx+dudPeTxx/params["PeM"]).all())
#Check ddvdPeMdt
assert(np.isclose(dvdPeTComputed,(-dvdPeTx+dvdPeTxx/params["PeT"]-vxx/(params["PeT"]**2)-params["delta"]*dvdPeT)/params["Le"]).all())

#Case 2: Nonlinear Source Portion
params={"PeM": 200, "PeT": 100, "f": .4, "Le": 2, "Da": .2, "beta": 10, "gamma": 1.1,"delta": 2, "vH": 4}
#Define nElem=1,nColl=2 model with roots at -1 and 1 for simplicity
model = TankModel(nCollocation=2,nElements=2,spacing="legendre",bounds=[0,1],params=params)
x=model.collocationPoints
#Case 1a: u=x^2, uPeM=x^2, v=x^2, vPeM=x^2
u = -x**3-x**2+5*x+5/params["PeM"]
v = lambda x:-x**2+2*x+(2/params["PeT"]+params["f"]*(2-1))/(1-params["f"])
vxx=-2 + x*0
dudPeT = -x**3-x**2+5*x+5/params["PeM"]
dudPeTx = -3*x**2-2*x+5
dudPeTxx = -6*x-2
dvdPeT = -x**3-x**2+5*x+((5+params["f"]*v(1)-v(0))/params["PeT"]+params["f"]*(5-2))/(1-params["f"])
dvdPeTx = -3*x**2-2*x+5
dvdPeTxx = -6*x-2
v=v(x)
y=np.concatenate((u,v,dudPeT,dvdPeT),axis=0)
dydt = model.dydtSens(y,0,paramSelect=["PeT"])
dudPeTComputed = dydt[2*model.nCollocation*model.nElements:3*model.nCollocation*model.nElements]
dvdPeTComputed = dydt[3*model.nCollocation*model.nElements:]
#Check ddudPeTdt
assert(np.isclose(dudPeTComputed, -dudPeTx+dudPeTxx/params["PeM"]\
                                    +params["Da"]*np.exp(params["gamma"]*params["beta"]*v/(1+params["beta"]*v))\
                                    *((1-u)*(params["gamma"]*params["beta"]*dvdPeT/(1+params["beta"]*v)**2)-dudPeT)).all())
#Check ddvdPeTdt
assert(np.isclose(dvdPeTComputed,(-dvdPeTx+dvdPeTxx/params["PeT"]-vxx/(params["PeT"]**2)-params["delta"]*dvdPeT\
                                  +params["Da"]*np.exp(params["gamma"]*params["beta"]*v/(1+params["beta"]*v))\
                                    *((1-u)*(params["gamma"]*params["beta"]*dvdPeT/(1+params["beta"]*v)**2)-dudPeT))/params["Le"]).all())
print("     PeT Sensitivity Passes")

#========================================== f Tests ==================================================================
print("     Testing f")
#Case 1: Transport and Diffusion Portion
#Set Da=0 so the nonlinear source term is 0
params={"PeM": 30, "PeT": 20, "f": 0, "Le": 3, "Da": 0, "beta": 0, "gamma": 0,"delta": 2, "vH":1}
#Define nElem=1,nColl=2 model with roots at -1 and 1 for simplicity
model = TankModel(nCollocation=2,nElements=2,spacing="legendre",bounds=[0,1],params=params)
x=model.collocationPoints
#Case 1a: u=x^2, uvH=x^2, v=x^2, vvH=x^2
u = -x**3-x**2+5*x+5/params["PeM"]
v = lambda x:-x**2+2*x+(2/params["PeT"]+params["f"]*(2-1))/(1-params["f"])
vxx=-2 + x*0
dudf = -x**3-x**2+5*x+5/params["PeM"]
dudfx = -3*x**2-2*x+5
dudfxx = -6*x-2
dvdf = -x**3-x**2+5*x+(5/params["PeT"]+params["f"]*(5-2)+v(1))/(1-params["f"])
dvdfx = -3*x**2-2*x+5
dvdfxx = -6*x-2
y=np.concatenate((u,v(x),dudf,dvdf),axis=0)
dydt = model.dydtSens(y,0,paramSelect=["f"])
dudfComputed = dydt[2*model.nCollocation*model.nElements:3*model.nCollocation*model.nElements]
dvdfComputed = dydt[3*model.nCollocation*model.nElements:]
#Check ddudPeMdt
assert(np.isclose(dudfComputed, -dudfx+dudfxx/params["PeM"]).all())
#Check ddvdPeMdt
assert(np.isclose(dvdfComputed,(-dvdfx+dvdfxx/params["PeT"]-params["delta"]*dvdf)/params["Le"]).all())

#Case 2: Nonlinear Source Portion
params={"PeM": 200, "PeT": 100, "f": .4, "Le": 2, "Da": .15, "beta": 10, "gamma": 1.1,"delta": 2, "vH": 4}
#Define nElem=1,nColl=2 model with roots at -1 and 1 for simplicity
model = TankModel(nCollocation=2,nElements=2,spacing="legendre",bounds=[0,1],params=params)
x=model.collocationPoints
#Case 1a: u=x^2, uPeM=x^2, v=x^2, vPeM=x^2
u = -x**3-x**2+5*x+5/params["PeM"]
v = lambda x:-x**2+2*x+(2/params["PeT"]+params["f"]*(2-1))/(1-params["f"])
vxx=-2 + x*0
dudf = -x**3-x**2+5*x+5/params["PeM"]
dudfx = -3*x**2-2*x+5
dudfxx = -6*x-2
dvdf = -x**3-x**2+5*x+(5/params["PeT"]+params["f"]*(5-2)+v(1))/(1-params["f"])
dvdfx = -3*x**2-2*x+5
dvdfxx = -6*x-2
v=v(x)
y=np.concatenate((u,v,dudf,dvdf),axis=0)
dydt = model.dydtSens(y,0,paramSelect=["f"])
dudfComputed = dydt[2*model.nCollocation*model.nElements:3*model.nCollocation*model.nElements]
dvdfComputed = dydt[3*model.nCollocation*model.nElements:]
#Check ddudfdt
assert(np.isclose(dudfComputed, -dudfx+dudfxx/params["PeM"]\
                                    +params["Da"]*np.exp(params["gamma"]*params["beta"]*v/(1+params["beta"]*v))\
                                    *((1-u)*(params["gamma"]*params["beta"]*dvdf/(1+params["beta"]*v)**2)-dudf)).all())
#Check ddvdfdt
assert(np.isclose(dvdfComputed,(-dvdfx+dvdfxx/params["PeT"]-params["delta"]*dvdf\
                                  +params["Da"]*np.exp(params["gamma"]*params["beta"]*v/(1+params["beta"]*v))\
                                    *((1-u)*(params["gamma"]*params["beta"]*dvdf/(1+params["beta"]*v)**2)-dudf))/params["Le"]).all())
print("     f Sensitivity Passes")

