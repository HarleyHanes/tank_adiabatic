import element 
import numpy as np
import matplotlib.pyplot as plt

# Test correct entering

testElement = element.Element(bounds=[-1,1],order=4,spacing="legendre")
print(testElement.order)
print(testElement.spacing)
print(testElement.bounds)
print(testElement.collocationPoints)

x=np.linspace(testElement.bounds[0]+.01,testElement.bounds[1]-.01,100)

basisValue = testElement.basisFunctions(x)
firstDeriv = testElement.basisFirstDeriv(x)
secondDeriv = testElement.basisSecondDeriv(x)

plt.figure
plt.plot(x,basisValue.transpose())
plt.title("Basis Functions")
plt.show()
plt.figure
plt.plot(x,firstDeriv.transpose())
plt.title("Basis First Derivatives")
plt.show()
plt.figure
plt.plot(x,secondDeriv.transpose())
plt.title("Basis Second Derivatives")
plt.show()