import sys
import os 
current_script_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath(os.path.join(current_script_dir,'..','..'))
sys.path.append(grandparent_dir)
print(sys.path)

import numpy as np
from tankModel.TankModel import TankModel
import scipy

print("Running testTankModel.py")
params={"PeM": 1, "PeT": 1, "f": 1, "Le": 1, "Da": 1, "beta": 1, "gamma": 1,"delta": 1, "vH":1}
model = TankModel(nCollocation=1,nElements=2,spacing="legendre",bounds=[-2,2],params=params)

trueMassBoundaryMat=np.array([[1+(3/2)/params["PeM"], -2/params["PeM"], 1/2/params["PeM"], 0 ,0],
                              [0, 1, 0, 0, 0],
                              [1/2, -2, 3, -2, 1/2],
                              [0, 0, 0, 1, 0],
                              [0, 0, 1/2, -2, 3/2]
                              ])
trueTempBoundaryMat=np.array([[1+(3/2)/params["PeT"], -2/params["PeT"], 1/2/params["PeT"], 0 ,-params["f"]],
                              [0, 1, 0, 0, 0],
                              [1/2, -2, 3, -2, 1/2],
                              [0, 0, 0, 1, 0],
                              [0, 0, 1/2, -2, 3/2]
                              ])

#Check Boundary matrices
assert(np.isclose(trueMassBoundaryMat, model.massBoundaryMat).all())
assert(np.isclose(trueTempBoundaryMat, model.tempBoundaryMat).all())
#Check first and 2nd order matrices
'''
Tasks
    1) Write unit tests for 1st and 2nd order matrices and 1 and 2 collocation points/ elements.
       Not expecting these to be the problem but it should be a quick check
       
'''

model = TankModel(nCollocation=2,nElements=2,spacing="uniform",bounds=[-3,3],params=params)


x = model.collocationPoints
l10=-1/6*(x**3+3*x**2+2*x)
l10x=-1/6*(3*x**2+6*x+2)
l10xx=-(x+1)
l11=1/2*(x**3+4*x**2+3*x)
l11x=1/2*(3*x**2+8*x+3)
l11xx=3*x+4
l12=-1/2*(x**3+5*x**2+6*x)
l12x=-1/2*(3*x**2+10*x+6)
l12xx=-3*x-5
l13=1/6*(x**3+6*x**2+11*x+6)
l13x=1/6*(3*x**2+12*x+11)
l13xx=x+2
l20=-1/6*(x**3-6*x**2+11*x-6)
l20x=-1/6*(3*x**2-12*x+11)
l20xx=2-x
l21=1/2*(x**3-5*x**2+6*x)
l21x=1/2*(3*x**2-10*x+6)
l21xx=3*x-5
l22=-1/2*(x**3-4*x**2+3*x)
l22x=-1/2*(3*x**2-8*x+3)
l22xx=4-3*x
l23=1/6*(x**3-3*x**2+2*x)
l23x=1/6*(3*x**2-6*x+2)
l23xx=x-1

#
expectedFirstOrderMat=np.zeros((4,7))    
expectedFirstOrderMat[0,0:4]=np.array([l10x[0], l11x[0],l12x[0],l13x[0]])      
expectedFirstOrderMat[1,0:4]=np.array([l10x[1], l11x[1],l12x[1],l13x[1]])       
expectedFirstOrderMat[2,3:7]=np.array([l20x[2], l21x[2],l22x[2],l23x[2]])       
expectedFirstOrderMat[3,3:7]=np.array([l20x[3], l21x[3],l22x[3],l23x[3]])    
assert(np.isclose(expectedFirstOrderMat, model.firstOrderMat).all())
expectedSecondOrderMat=np.zeros((4,7))    
expectedSecondOrderMat[0,0:4]=np.array([l10xx[0], l11xx[0],l12xx[0],l13xx[0]])      
expectedSecondOrderMat[1,0:4]=np.array([l10xx[1], l11xx[1],l12xx[1],l13xx[1]])       
expectedSecondOrderMat[2,3:7]=np.array([l20xx[2], l21xx[2],l22xx[2],l23xx[2]])       
expectedSecondOrderMat[3,3:7]=np.array([l20xx[3], l21xx[3],l22xx[3],l23xx[3]]) 
assert(np.isclose(expectedSecondOrderMat, model.secondOrderMat).all())


#Check dydt computed values
params={"PeM": 1, "PeT": 1, "f":0, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 0, "vH":0}
#Define nElem=1,nColl=2 model with roots at -1 and 1 for simplicity
#model = TankModel(nCollocation=2,nElements=1,spacing="legendre",bounds=[-np.sqrt(3),np.sqrt(3)],params=params)
#If use a [0,1] bound, coeffecients are 1-off
#model = TankModel(nCollocation=3,nElements=1,spacing="legendre",bounds=[-np.sqrt(5/3),np.sqrt(5/3)],params=params)
model = TankModel(nCollocation=5,nElements=1,spacing="legendre",bounds=[0,1],params=params)
'''
Notes on unit testing results
If bounds are [0,1] 2+ collocation pointeswith params so u and v should be identical
    2) Computed dvdt are not the same as for dudt. Ceoffecients are much larger
    3) As PeM and PeT increase (less diffusion), error gets smaller and the computed v converge to u
'''

x=model.collocationPoints
print(x)
nPoints=model.nCollocation*model.nElements
#Case 1a: u=x^2, uvH=x^2, v=x^2, vvH=x^2
order=3
if order==2:
    u = -x**2+2*x+2/params["PeM"]
    v = -x**2+2*x+(2/params["PeT"]+params["f"]*(2-1))/(1-params["f"])
    dudx = -2*x+2
    dvdx = -2*x+2
    d2udx2 = -2
    d2vdx2 = -2
elif order==3:
    u = -x**3-x**2+5*x+5/params["PeT"]
    v = -x**3-x**2+5*x+(5/params["PeT"]+params["f"]*(5-2))/(1-params["f"])
    dudx = -3*x**2-2*x+5
    dvdx = -3*x**2-2*x+5
    d2udx2 = -6*x-2
    d2vdx2 = -6*x-2
    
if params["f"]==1:
    v=-x**2+2*x+1
else:
    v = -x**3-x**2+5*x+(5/params["PeT"]+params["f"]*(5-2))/(1-params["f"])
y=np.concatenate((u,v),axis=0)
dydt = model.dydt(y,0)
#print(model.massFullCoeffMat)

print("u Computed", dydt[:nPoints])
print("u Expected", -dudx+d2udx2/params["PeM"]+params["Da"]*(1-u)*np.exp(params["gamma"]*params["beta"]/(1+params["beta"]*v)))
print("v Computed: ", dydt[nPoints:])
print("v Expected: ",(-dvdx+d2vdx2/params["PeT"]+params["Da"]*(1-u)\
                            *np.exp(params["gamma"]*params["beta"]/(1+params["beta"]*v))\
                            +params["delta"]*(params["vH"]-v))/params["Le"])
assert(np.isclose(dydt[:nPoints],-dudx+d2udx2/params["PeM"]+params["Da"]*(1-u)\
                            *np.exp(params["gamma"]*params["beta"]/(1+params["beta"]*v))).all())
assert(np.isclose(dydt[nPoints:],(-dvdx+d2vdx2/params["PeT"]+params["Da"]*(1-u)\
                            *np.exp(params["gamma"]*params["beta"]/(1+params["beta"]*v))\
                            +params["delta"]*(params["vH"]-v))/params["Le"]).all())



#Check Integration: Spatial Integration Only
nCollocation=2
nElements=2
model = TankModel(nCollocation=nCollocation,nElements=nElements,spacing="legendre",bounds=[-2,2],params=params)
f=lambda x: x**2+x+1
fint = 9+1/3
integral=model.integrateSpace(f)

nCollocation=3
nElements=2
model = TankModel(nCollocation=nCollocation,nElements=nElements,spacing="legendre",bounds=[0,2],params=params)
f=lambda x: x**3+x**2+x+1
fint = 10+2/3
integral=model.integrateSpace(f)
assert(np.isclose(fint,integral))

#Check Integration: Spatial at multiple points and temporal
nCollocation=2
nElements=2
model = TankModel(nCollocation=nCollocation,nElements=nElements,spacing="legendre",bounds=[-2,2],params=params)
f=lambda x: np.outer(np.array([0,1,2]),x**2+x+1).flatten()
fint = np.array([0,9+1/3, 18+2/3])
fTempint = 18+2/3
integral,integralSpace=model.integrate(f,np.array([0,1,2]))
assert(np.isclose(fTempint,integral))
assert(np.isclose(fint,integralSpace).all())

#Check Integration: Spatial and Temporal Integration
print("testTankModely.py passes")