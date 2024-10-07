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
nCollocations = [3]
verbosity = 1

#I think there's an error with the higher
higherOrderTerms=["sin"]  #Must be greater than 2 to satisfy BC
nElems = np.array([2,4,8,16,32,64,128])  #Cant use nElems=1 due to some dimensionality issues with squeeze
parameterSet="Bizon2012_stable"
#Parameter limitations:
# Non-negative: Da, gamma, beta, delta
# Positive: Le, PeM,
# Other: f can be any number except 1
# Parameter Relations
# If delta=0: then vH has no effect 
# If Da=0: then beta and gamma have no effect
# If Da=0 and delta=0: then Le has no effect

#Parameter set with improved conditioning
if parameterSet=="Bizon2012_improvedConditioning":
    params={"PeM": 1000, "PeT": 1000, "f": .3, "Le": 1, "Da": .15, "beta": 1.4, "gamma": 10,"delta": 2, "vH":-.2}
#Bizon2012 Parameters for  a stable domain
elif parameterSet=="Bizon2012_stable":
    params={"PeM": 300, "PeT": 300, "f": .3, "Le": 1, "Da": .15, "beta": 1.4, "gamma": 10,"delta": 2, "vH":-.2}
#Bizon2012 Parameters without nonlinear effects
elif parameterSet=="Bizon2012_linear":
    params={"PeM": 300, "PeT": 300, "f": .3, "Le": 1, "Da": 0, "beta": 0, "gamma": 0, "delta": 2, "vH": -.2}
#Bizon2012 Parameters with just diffusion/advection
elif parameterSet=="Bizon2012_diffAdvec":
    params={"PeM": 300, "PeT": 300, "f": 0, "Le": 1, "Da": 0, "beta": 0, "gamma": 0, "delta": 0, "vH": 0}
#Unit parameters with just diffusion/advection
#params={"PeM": 1, "PeT": 1, "f": .5, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 0, "vH": 0}

resultsFolder = "../../results/verification/"
tEval = np.linspace(0,3,5)
xEval = np.linspace(0,1,100)
error, solutions, convergence, errorSpace, convergenceSpace=tankMMS.runMMStest(higherOrderTerms,nCollocations,nElems,xEval,tEval,params,verbosity=verbosity)

for iOrder in range(len(higherOrderTerms)):
    print("Spatial Order: " + str(higherOrderTerms[iOrder]))
    for iColl in range(len(nCollocations)):
        print("    Collocation Points: " + str(nCollocations[iColl]))
        print("        Time Constant Discretiziation")
        print("            u L2 Errors: " + str(errorSpace[iColl,:,0,iOrder,0,0,0]))
        print("            u Linf Errors: " + str(errorSpace[iColl,:,0,iOrder,0,1,0]))
        print("            v L2 Errors: " + str(errorSpace[iColl,:,0,iOrder,1,0,0]))
        print("            v Linf Errors: " + str(errorSpace[iColl,:,0,iOrder,1,1,0]))
        print("            u L2 Convergence: " + str(convergenceSpace[iColl,:,0,iOrder,0,0,0]))
        print("            u Linf Convergence: " + str(convergenceSpace[iColl,:,0,iOrder,0,1,0]))
        print("            v L2 Convergence: " + str(convergenceSpace[iColl,:,0,iOrder,1,0,0]))
        print("            v Linf Convergence: " + str(convergenceSpace[iColl,:,0,iOrder,1,1,0]))
        for itemporal in range(3):
            print("        Temporal " + str(itemporal))
            print("            u L2 Errors: " + str(error[iColl,:,itemporal,iOrder,0,0]))
            print("            u Linf Errors: " + str(error[iColl,:,itemporal,iOrder,0,1]))
            print("            v L2 Errors: " + str(error[iColl,:,itemporal,iOrder,1,0]))
            print("            v Linf Errors: " + str(error[iColl,:,itemporal,iOrder,1,1]))
            print("            u L2 Convergence: " + str(convergence[iColl,:,itemporal,iOrder,0,0]))
            print("            u Linf Convergence: " + str(convergence[iColl,:,itemporal,iOrder,0,1]))
            print("            v L2 Convergence: " + str(convergence[iColl,:,itemporal,iOrder,1,0]))
            print("            v Linf Convergence: " + str(convergence[iColl,:,itemporal,iOrder,1,1]))

    # Assess results at t=0 (no time variation considered, just spatial discretization)
    #Compare expected and computed results in the time-constant case

    saveLocation = resultsFolder+"/"+parameterSet+"/nCol_"+str(nCollocations[iColl])+"_order"+str(higherOrderTerms[iOrder])
    plt.figure()
    plt.plot(xEval,solutions[iColl,0,0,iOrder,0,0,0,:])
    plt.plot(xEval,solutions[iColl,0,0,iOrder,1,0,0,:])
    plt.legend(["u","v"])
    plt.title("Manufactured solution for u and v at t=0")

    #Plot the computed solution for each discretization level at t=0
    plt.figure()
    plt.plot(xEval,solutions[iColl,:,0,iOrder,0,1,0,:].transpose())
    plt.title("Computed Solutions for u at t=0")
    plt.legend(['nElem='+str(nElement) for nElement in nElems])

    
    plt.figure()
    plt.plot(xEval,solutions[iColl,:,0,iOrder,1,1,0,:].transpose())
    plt.title("Computed Solutions for v at t=0")
    plt.legend(['nElem='+str(nElement) for nElement in nElems])
    #print(solutions[iColl,:,0,iOrder,1,1,0,:].transpose())

    #Plot the error and expected convergence rate of the spatial discretization
    expectedRate=nCollocations[iColl]+2
    convergenceExampleStartValue=np.min(errorSpace[iColl,0,0,iOrder,:,:,0])*.35
    convergenceExample=convergenceExampleStartValue*(nElems[0]/nElems)**(expectedRate)
    plt.figure()
    plt.loglog(nElems, errorSpace[iColl,:,0,iOrder,0,0,0])
    plt.loglog(nElems, errorSpace[iColl,:,0,iOrder,0,1,0])
    plt.loglog(nElems, errorSpace[iColl,:,0,iOrder,1,0,0])
    plt.loglog(nElems, errorSpace[iColl,:,0,iOrder,1,1,0])
    plt.loglog(nElems, convergenceExample)
    plt.xlabel(r"$N_E$")
    plt.ylabel("Relative Error")
    plt.legend([r"u, $L^2$ Error", r"u, $L^\inf$ Error",r"v, $L^2$ Error", r"v, $L^\inf$ Error", r"Order " +str(expectedRate)+ r" Convergence"])
    plt.title("Solution Convergence of Spatial Discretization")
    if not os.path.exists(saveLocation):
        os.makedirs(saveLocation)
    plt.savefig(saveLocation+"/spatialConvergence.pdf",format='pdf')
    plt.savefig(saveLocation+"/spatialConvergence.png",format='png')


    #Plot the error and expected convergence rate of the joint space/temporal discretization

    for itemporal in range(3):
        expectedRate=nCollocations[iColl]+2
        convergenceExampleStartValue=np.min(error[iColl,0,itemporal,iOrder,:,:])*.35
        convergenceExample=convergenceExampleStartValue*(nElems[0]/nElems)**(expectedRate)
        plt.figure()
        plt.loglog(nElems, error[iColl,:,itemporal,iOrder,0,0])
        plt.loglog(nElems, error[iColl,:,itemporal,iOrder,0,1])
        plt.loglog(nElems, error[iColl,:,itemporal,iOrder,1,0])
        plt.loglog(nElems, error[iColl,:,itemporal,iOrder,1,1])
        plt.loglog(nElems, convergenceExample)
        plt.xlabel(r"$N_E$")
        plt.ylabel("Relative Error")
        plt.legend([r"u, $L^2$ Error", r"u, $L^\inf$ Error",r"v, $L^2$ Error", r"v, $L^\inf$ Error", r"Order " +str(expectedRate)+ r" Convergence"])
        if itemporal==0:
            plt.title(r"Solution Convergence for $g(t)=1$")
        elif itemporal==1:
            plt.title(r"Solution Convergence for $g(t)=1+t$")
        elif itemporal==2:
            plt.title(r"Solution Convergence for $g(t)=1+te^{2t}$")
    
        plt.savefig(saveLocation+"/convergence_temporal"+str(itemporal)+".pdf",format='pdf')
        plt.savefig(saveLocation+"/convergence_temporal"+str(itemporal)+".png",format='png')
    
    plt.figure()
    plt.plot(xEval,solutions[iColl,0,0,iOrder,0,0,0,:]-solutions[iColl,0,0,iOrder,0,1,0,:])
    plt.plot(xEval,solutions[iColl,0,0,iOrder,1,0,0,:]-solutions[iColl,0,0,iOrder,1,1,0,:])
    plt.legend(["u","v"])
    plt.title("Pointwise Error for u and v at t=0")
    
    plt.show()
    # #Plot the pointwise error of each as discretization is refined
    # for iCol in range(len(nCollocations)):
    #     for iOrder in range(len(higherOrderTerms)):
    #         plt.figure()
    #         for iElem in range(len(nElems)):
    #             plt.semilogy(xEval,np.abs((solutions[iCol,iElem,0,iOrder,0,1,0,:]-solutions[iCol,iElem,0,iOrder,0,0,0,:])/solutions[iCol,iElem,0,iOrder,0,0,0,:]))
    #         plt.title("Error of uMMS at t start for each discretization")
    #         plt.legend(['nElem='+str(nElement) for nElement in nElems])
    #         plt.xlabel('x')
    #         plt.ylabel(r"$\frac{|u_{Model}-u_{Computed}|}{|u_{Model}|}$")

    # #Plot the pointwise error of each as discretization is refined
    # for iCol in range(len(nCollocations)):
    #     for iOrder in range(len(higherOrderTerms)):
    #         plt.figure()
    #         for iElem in range(len(nElems)):
    #             plt.semilogy(xEval,np.abs((solutions[iCol,iElem,0,iOrder,1,1,0,:]-solutions[iCol,iElem,0,iOrder,1,0,0,:])/solutions[iCol,iElem,0,iOrder,1,0,0,:]))
    #         plt.title("Error of vMMS at t start for each discretization")
    #         plt.legend(['nElem='+str(nElement) for nElement in nElems])
    #         plt.xlabel('x')
    #         plt.ylabel(r"$\frac{|v_{Model}-u_{Computed}|}{|v_{Model}|}$")
    #Assess time-variant results
    #Plot the computed solution for each discretization level at t=end
    plt.figure()
    plt.plot(xEval,solutions[iColl,:,0,iOrder,0,1,-1,:].transpose())
    plt.title("Computed Solutions for u at t end")
    plt.legend(['nElem='+str(nElement) for nElement in nElems])

    #Plot the error of each as discretization is refined
    for iCol in range(len(nCollocations)):
        for iOrder in range(len(higherOrderTerms)):
            plt.figure()
            for iElem in range(len(nElems)):
                plt.semilogy(xEval,np.abs((solutions[iCol,iElem,0,iOrder,0,1,-1,:]-solutions[iCol,iElem,0,iOrder,0,0,-1,:])/solutions[iCol,iElem,0,iOrder,0,1,-1,:]))
            plt.title("Error of uMMS at t end for each discretization")
            plt.legend(['nElem='+str(nElement) for nElement in nElems])
            plt.xlabel('x')
            plt.ylabel(r"$\frac{|u_{Model}-u_{Computed}|}{|u_{Model}|}$")


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
    
