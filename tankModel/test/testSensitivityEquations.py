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
params={"PeM": 1, "PeT": 1, "f": 1, "Le": 1, "Da": 0, "beta": 1, "gamma": 1,"delta": 1, "vH":1}
#Define nElem=1,nColl=2 model with roots at -1 and 1 for simplicity
model = TankModel(nCollocation=2,nElements=1,spacing="legendre",bounds=[-np.sqrt(3),np.sqrt(3)],params=params)
x=model.collocationPoints
#Case 1a: u=x^2, uvH=x^2, v=x^2, vvH=x^2
u = lambda x:-x**2+x+1/params["PeM"]
v = lambda x:-x**2+x+1/(1-params["f"])*(1/params["PeT"]+params["f"]())
y=np.concatenate((x**2,x**2,x**2,x**2),axis=0)
print(y)
dydt = model.dydtSens(y,0,paramSelect=["vH"])
#Check ddudvHdt
print(dydt)
print(-2*x+2/params["PeM"])
assert(np.isclose(dydt[4:6],-2*x+2/params["PeM"]).all())
#Check ddvdvHdt
assert(np.isclose(dydt[6:],1/params["Le"]*(-2*x+2/params["PeT"])+params["delta"]-params["vH"]*(x**2)).all())

params={"PeM": 7, "PeT": 31, "f": 1, "Le": 1, "Da": 0, "beta": 1, "gamma": 1,"delta": 8, "vH":9}
#Define nElem=1,nColl=2 model with roots at -1 and 1 for simplicity
model = TankModel(nCollocation=2,nElements=1,spacing="legendre",bounds=[-np.sqrt(3),np.sqrt(3)],params=params)
dydt = model.dydtSens(y,0,paramsSelect=["vH"])
#Check ddudvHdt
assert(np.isclose(dydt[4:6],-2*x+2/params["PeM"]).all())
#Check ddvdvHdt
assert(np.isclose(dydt[6:],1/params["Le"]*(-2*x+2/params["PeT"])+params["delta"]-params["vH"]*(x**2)).all())
print("         vH Transport and Diffusion Portion Passes")

#Case 2: Nonlinear Source Portion
#Can't set transport terms to 0 due to BC but will use linear terms so diffusion term is 0
params={"PeM": 1, "PeT": 1, "f": 1, "Le": 1, "Da": 1, "beta": 1, "gamma": 1,"delta": 1, "vH":1}
#Define nElem=1,nColl=2 model 
model = TankModel(nCollocation=2,nElements=1,spacing="legendre",bounds=[0,1],params=params)
x=model.collocationPoints
#Case 2a: u=1, uvH=x, v=x, vvH=x
y=np.append(1+x*0,x,x,x)
dydt = model.dydtSens(y,0,paramSelect=["vH"])
#Check ddudvHdt
assert(np.isclose(dydt[4:6],-1-params["Da"]*np.exp(params["gamma"]*params["beta"]*x/(1+params["beta"]*x))*x).all())
#Check ddvdvHdt
assert(np.isclose(dydt[6:],1/params["Le"]*(-1-params["Da"]*np.exp(params["gamma"]*params["beta"]*x/(1+params["beta"]*x))*x \
                                           + params["delta"]-params["vH"]*x)).all())

#Case2b: u =x, uvH=0, v=x^2, vvH=x
y=np.append(x,x*0,x**2,x)
dydt = model.dydtSens(y,0,paramSelect=["vH"])
#Check ddudvHdt
assert(np.isclose(dydt[4:6],-1-params["Da"]*np.exp(params["gamma"]*params["beta"]*x**2/(1+params["beta"]*x**2))\
                                *(1-x)*(params["gamma"]*params["beta"]*(1+2*params["beta"]*x**3)/(1+params["beta"]*x**2)**2)).all())
#Check ddvdvHdt
assert(np.isclose(dydt[6:],1/params["Le"]*(-1-params["Da"]*np.exp(params["gamma"]*params["beta"]*x**2/(1+params["beta"]*x**2))\
                                *(1-x)*(params["gamma"]*params["beta"]*(1+2*params["beta"]*x**3)/(1+params["beta"]*x**2)**2)\
                                +params["delta"]-params["vH"]*x)).all())
