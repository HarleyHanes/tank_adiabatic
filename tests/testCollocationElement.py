import CollocationElement 
import numpy as np
import matplotlib.pyplot as plt

# Test correct entering

testElement = CollocationElement.Element(bounds=[0,3],nCollocation=2,spacing="uniform")

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


testElement = CollocationElement.Element(bounds=[.56,1.98],nCollocation=1,spacing="legendre")

f = lambda x: 12.67*x+5.34
fint = 30.431878

assert(np.isclose(testElement.integrate(f),fint))


testElement = CollocationElement.Element(bounds=[.5,4.5],nCollocation=2,spacing="legendre")

f = lambda x: 2*(x**3)+3.4*(x**2)+12*x+5
fint = 448.133356

assert(np.isclose(testElement.integrate(f),fint))


testElement = CollocationElement.Element(bounds=[-2,-.5],nCollocation=3,spacing="legendre")

f = lambda x: .5*(x**5)+x**4+2*(x**3)+3*(x**2)+4*x+5
fint = 0.96797

assert(np.isclose(testElement.integrate(f),fint))


print("testsElement.py Passes")
