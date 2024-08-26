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
nCollocations = [2]
verbosity = 2

#I think there's an error with the higher
spatialOrders=[2,3,4]  #Must be greater than 2 to satisfy BC
nElems = [2,4,8,16]  #Cant use nElems=1 due to some dimensionality issues with squeeze
#Note: Parameters can be any positive value except f=1
params={"PeM": 1, "PeT": 1, "f": 2, "Le": 1, "Da": 1, "beta": 1, "gamma": 1,"delta": 1, "vH":1}
tEval = np.linspace(0,3,200)
xEval = np.linspace(0,1,200)
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
    plt.plot(xEval,xEval**2-2*xEval-2)
    plt.title("True manufactured solution for u")
    plt.show()

    plt.plot(xEval,solutions[iColl,:,0,iOrder,0,0,-1,:].transpose())
    plt.title("Computed manufactured Solutions for u")
    plt.show()

    #Plot the computed solution for each element
    plt.plot(xEval,solutions[iColl,:,0,iOrder,0,1,-1,:].transpose())
    plt.title("Computed Solutions for u")
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
#    Forgot to divide whole temp equation by Le, doesn't effect current cases since Le=1
    
