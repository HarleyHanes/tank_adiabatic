import element 
import numpy as np
import matplotlib.pyplot as plt

# Test correct entering

testElement = element.Element(bounds=[0,3],order=2,spacing="uniform")

x=np.linspace(testElement.bounds[0],testElement.bounds[1],10)

assert(np.isclose(testElement.interpolationPoints,np.array([0,1,2,3])).all())

trueBasis=np.array([-1/6*(x**3-6*x**2+11*x-6),
                      1/2*(x**3-5*x**2+6*x),
                     -1/2*(x**3-4*x**2+3*x),
                      1/6*(x**3-3*x**2+2*x)])
trueFirstDeriv=np.array([-1/6*(3*x**2-12*x+11),
                          1/2*(3*x**2-10*x+6),
                         -1/2*(3*x**2-8*x+3),
                          1/6*(3*x**2-6*x+2)])
trueSecondDeriv=np.array([-1/6*(6*x-12),
                           1/2*(6*x-10),
                          -1/2*(6*x-8),
                           1/6*(6*x-6)])

assert(np.isclose(trueBasis,testElement.basisFunctions(x)).all())
assert(np.isclose(trueFirstDeriv,testElement.basisFirstDeriv(x)).all())
assert(np.isclose(trueSecondDeriv,testElement.basisSecondDeriv(x)).all())
print("testsElement.py Passes")