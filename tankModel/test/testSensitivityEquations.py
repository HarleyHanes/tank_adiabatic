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
model = TankModel(nCollocation=2,nElements=2,spacing="legendre",bounds=[0,1],params=params)
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
dudvHtComputed = dydt[2*model.nCollocation*model.nElements:3*model.nCollocation*model.nElements]
dvdvHtComputed = dydt[3*model.nCollocation*model.nElements:]
#Check ddudvHdt
assert(np.isclose(dudvHtComputed, -dudvHx+dudvHxx/params["PeM"]                                        ).all())
#Check ddvdvHdt
assert(np.isclose(dvdvHtComputed,(-dvdvHx+dvdvHxx/params["PeT"]+params["delta"]*(1-dvdvH))/params["Le"]).all())

#Case 2: Nonlinear Source Portion
params={"PeM": 1, "PeT": 1, "f": 0, "Le": 1, "Da": 1, "beta": 1, "gamma": 1,"delta": 0, "vH": 0}
#Define nElem=1,nColl=2 model with roots at -1 and 1 for simplicity
model = TankModel(nCollocation=1,nElements=2,spacing="legendre",bounds=[0,1],params=params)
x=model.collocationPoints
#Case 1a: u=x^2, uvH=x^2, v=x^2, vvH=x^2
#u = -x**2+2*x+2/params["PeM"]
u=0*x
v = -x**2+2*x+(2/params["PeT"]+params["f"]*(2-1))/(1-params["f"])
#dudvH = -x**2+2*x+2/params["PeM"]
dudvH=0*x
#dvdvH = -x**2+2*x+(2/params["PeT"]+params["f"]*(2-1))/(1-params["f"])
dudvHx=0*x
dudvHxx=0*x
#dudvHx = -2*x+2
#dvdvHx = -2*x+2
#dudvHxx = -2
#dvdvHxx = -2
dvdvH=0*x
dvdvHx=0*x
dvdvHxx=0*x
y=np.concatenate((u,v,dudvH,dvdvH),axis=0)
print(y)
dydt = model.dydtSens(y,0,paramSelect=["vH"])
dudvHtComputed = dydt[2*model.nCollocation*model.nElements:3*model.nCollocation*model.nElements]
dvdvHtComputed = dydt[3*model.nCollocation*model.nElements:]
print(dvdvHtComputed)
print(1/((-x**2+2*x+3)**2)*np.exp((-x**2+2*x+2)/(-x**2+2*x+3)))
print((-dvdvHx+dvdvHxx/params["PeT"]\
                                  +params["Da"]*np.exp(params["gamma"]*params["beta"]*v/(1+params["beta"]*v))\
                                    *((1-u)*(params["gamma"]*params["beta"]*(1+2*params["beta"]*v*dvdvH)/(1+params["beta"]*v)**2)-dudvH)\
                                    +params["delta"]*(1-dvdvH))/params["Le"])
#Check ddudvHdt
# assert(np.isclose(dudvHtComputed, -dudvHx+dudvHxx/params["PeM"]\
#                                     +params["Da"]*np.exp(params["gamma"]*params["beta"]*v/(1+params["beta"]*v))\
#                                     *((1-u)*(params["gamma"]*params["beta"]*(1+2*params["beta"]*v*dvdvH)/(1+params["beta"]*v)**2)-dudvH)).all())
#Check ddvdvHdt
assert(np.isclose(dvdvHtComputed,(-dvdvHx+dvdvHxx/params["PeT"]\
                                  +params["Da"]*np.exp(params["gamma"]*params["beta"]*v/(1+params["beta"]*v))\
                                    *((1-u)*(params["gamma"]*params["beta"]*(1+2*params["beta"]*v*dvdvH)/(1+params["beta"]*v)**2)-dudvH)\
                                    +params["delta"]*(1-dvdvH))/params["Le"]).all())
print("         vH Transport and Diffusion Portion Passes")