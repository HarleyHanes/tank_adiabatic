import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath(os.path.join(current_script_dir,'..',))
sys.path.append(grandparent_dir)
print(sys.path)

import numpy as np
import matplotlib.pyplot as plt
import scipy
import tankMMS
nCollocations = [1]
verbosity = 3

#I think there's an error with the higher
spatialOrders=[2]  #Must be greater than 2 to satisfy BC
nElems = [2]  #Cant use nElems=1 due to some dimensionality issues with squeeze
#Parameter limitations:
# Non-negative: Da, gamma, beta, delta
# Positive: Le, PeM,
# Other: f can be any number except 1
# Parameter Relations
# If delta=0: then vH has no effect 
# If Da=0: then beta and gamma have no effect
# If Da=0 and delta=0: then Le has no effect

#Bizon2012 Parameters for  a stable domain
#params={"PeM": 300, "PeT": 300, "f": .3, "Le": 1, "Da": .15, "beta": 1.4, "gamma": 10,"delta": 2, "vH":-.2}
#Bizon2012 Parameters with just diffusion/advection
#params={"PeM": 300, "PeT": 300, "f": 0, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 0, "vH": 0}
#Unit parameters with just diffusion/advection
params={"PeM": 10, "PeT": 10, "f": 0, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 0, "vH": 0}

tEval = np.linspace(0,3,50)
xEval = np.linspace(0,1,20)
error, solutions, convergence=tankMMS.runMMStest(spatialOrders,nCollocations,nElems,xEval,tEval,params,verbosity=verbosity)

for iOrder in range(len(spatialOrders)):
    print("Spatial Order: " + str(spatialOrders[iOrder]))
    for iColl in range(len(nCollocations)):
        print("    Collocation Points: " + str(nCollocations[iColl]))
        for itemporal in range(2):
            print("        Temporal " + str(itemporal))
            print("            u L2 Errors: " + str(error[iColl,:,itemporal,iOrder,0,0]))
            print("            u Linf Errors: " + str(error[iColl,:,itemporal,iOrder,0,1]))
            print("            v L2 Errors: " + str(error[iColl,:,itemporal,iOrder,1,0]))
            print("            v Linf Errors: " + str(error[iColl,:,itemporal,iOrder,1,1]))
            print("            u L2 Convergence: " + str(convergence[iColl,:,itemporal,iOrder,0,0]))
            print("            u Linf Convergence: " + str(convergence[iColl,:,itemporal,iOrder,0,1]))
            print("            v L2 Convergence: " + str(convergence[iColl,:,itemporal,iOrder,1,0]))
            print("            v Linf Convergence: " + str(convergence[iColl,:,itemporal,iOrder,1,1]))
    #Compare expected and computed results for u in the time-constant case
    plt.figure()
    plt.plot(xEval,-xEval**2+2*xEval+2)
    plt.title("True manufactured solution for u")
    
    plt.figure()
    plt.plot(xEval,solutions[iColl,:,0,iOrder,0,0,-1,:].transpose())
    plt.title("Computed manufactured Solutions for u")
    

    #Plot the computed solution for each element
    plt.figure()
    plt.plot(xEval,solutions[iColl,:,0,iOrder,0,1,-1,:].transpose())
    plt.title("Computed Solutions for u")

    #Plot the error of each as elements are refined
    for iCol in range(len(nCollocations)):
        for iOrder in range(len(spatialOrders)):
            plt.figure()
            for iElem in range(len(nElems)):
                plt.semilogy(xEval,np.abs(solutions[iCol,iElem,0,iOrder,0,1,0,:]-solutions[iCol,iElem,0,iOrder,0,0,0,:]))
            plt.title("Error of uMMS at t start for each discretization")
            plt.legend(['nElem='+str(element) for element in nElems])
            plt.xlabel('x')
            plt.ylabel('|u_{Model}-u_{Computed}|')
            plt.figure()
            for iElem in range(len(nElems)):
                plt.semilogy(xEval,np.abs(solutions[iCol,iElem,0,iOrder,0,1,-1,:]-solutions[iCol,iElem,0,iOrder,0,0,-1,:]))
            plt.title("Error of uMMS at t end for each discretization")
            plt.legend(['nElem='+str(element) for element in nElems])
            plt.xlabel('x')
            plt.ylabel('|u_{Model}-u_{Computed}|')
    plt.show()

## List of likely identified errors
# 1) Manufactured Solutions seem incorrect. 
#    They're not at all like the computed solutions but they also don't seem to satisfy BC (2nd order has non-zero deriv at x=1)
#    Higher order polynomials don't look like higher order polynomials
#    Comptued solutions do satisfy both these optical checks
# 2) Quadrature Rules are incorrect
#    Something's wrong with the trapezoid rule
# 3) Instability for nCollocation >2
#    Candidate Causes
#       Poor conditioning: RHS matrices often 1,000-10,000 cond numbers, however this also occurs for nCollocation=2
# 4) Incorrect ODE implemented (FIXED)
#    Forgot to divide whole temp equation by Le, doesn't affect current cases since Le=1
    
