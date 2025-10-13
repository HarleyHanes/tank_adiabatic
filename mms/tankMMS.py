import numpy as np
import scipy
from tankModel.TankModel import TankModel
import matplotlib.pyplot as plt


def runMMStest(higherOrders,nCollocations,nElems,xEval,tEval,params,verbosity = 0):
    #Parse inputs
    temporals=[lambda t: 1+0*t,lambda t: 1+1*t, lambda t: 1+t*np.exp(2*t)]
    temporalsdt=[lambda t: 0*t,lambda t: 1+0*t, lambda t: np.exp(2*t)+2*t*np.exp(2*t)]
    #temporals = [lambda t: 1+0*t, lambda t: t, lambda t: t**2, lambda t: np.sin(t)]
    #temporalsdt = [lambda t: 0*t, lambda t: 1+0*t, lambda t: 2*t, lambda t: np.cos(t)]
    #Pre allocate solution and error arrays, the (2,2) indices are for (u,v) and then (MMS,Model) where MMS is true value and Model is computed value
    error = np.empty((len(nCollocations),len(nElems),len(temporals),len(higherOrders),2,2))
    errorSpace = np.empty((len(nCollocations),len(nElems),len(temporals),len(higherOrders),2,2,tEval.size))
    solutions= np.empty((len(nCollocations),len(nElems),len(temporals),len(higherOrders),2,2,tEval.size,xEval.size))
    jointConvergenceRates = np.empty((len(nCollocations),len(nElems)-1,len(temporals),len(higherOrders),2,2))
    spatialConvergenceRates = np.empty((len(nCollocations),len(nElems)-1,len(temporals),len(higherOrders),2,2,tEval.size))
    for iColl in range(len(nCollocations)):
        for iElem in range(len(nElems)):
            if verbosity > 0 :
                print("Testing for (collocation points, elements): (" + str(nCollocations[iColl])+ ", " + str(nElems[iElem])+ ")")
            # Generate tankModel
            model = TankModel(nCollocation=nCollocations[iColl],nElements=nElems[iElem],spacing="legendre",bounds=[0,1],params=params, verbosity =verbosity)
            #Read back params so that boundary Peclets are saved if entered params vector did not include them
            params=model.params
            # Set discretization
            x = model.collocationPoints
            #Loop through exact monomial cases
            #Loop through temporal functions
            for itemporal in range(len(temporals)):
                if verbosity > 0:
                    print("iTemporal: ", itemporal)
                for iorder in range(len(higherOrders)):
                    if type(higherOrders[iorder])==int:
                        spatialOrder=higherOrders[iorder]
                        if verbosity > 0:
                            print("Order: ", spatialOrder)
                        #Construct solutions that are sum of monomials up to spatialOrder 
                        u, dudt, dudx, dudx2, v, dvdt, dvdx, dvdx2 = constructPolynomialMMSsolutionFunction(spatialOrder,params,temporals[itemporal],temporalsdt[itemporal])
                    if type(higherOrders[iorder])==str:
                        if higherOrders[iorder]=="sin":
                            u, dudt, dudx, dudx2, v, dvdt, dvdx, dvdx2 = constructSinMMssolutionFunction(params,temporals[itemporal],temporalsdt[itemporal])
                        if higherOrders[iorder]=="nonSeperableSinusoidal":
                            u, dudt, dudx, dudx2, v, dvdt, dvdx, dvdx2 = constructNonSeperableMMSsolutionFunction(params,spatialOrder)
                    sourceFunction = constructSourceTermFunction(lambda t: u(t,model.collocationPoints), 
                                                                 lambda t: dudt(t,model.collocationPoints), 
                                                                 lambda t: dudx(t,model.collocationPoints), 
                                                                 lambda t: dudx2(t,model.collocationPoints), 
                                                                 lambda t: v(t,model.collocationPoints), 
                                                                 lambda t: dvdt(t,model.collocationPoints), 
                                                                 lambda t: dvdx(t,model.collocationPoints), 
                                                                 lambda t: dvdx2(t,model.collocationPoints),params)
                    
                    y0=np.append(u(tEval[0],model.collocationPoints),v(tEval[0],model.collocationPoints))
                    uErrorFunction = lambda x: model.eval(x,y0,output="u") - u(0,x)
                    vErrorFunction = lambda x: model.eval(x,y0,output="v")- v(0,x)
                    uSquaredErrorFunction = lambda x: uErrorFunction(x)**2
                    vSquaredErrorFunction = lambda x: vErrorFunction(x)**2
                    uSquaredReferenceFunction = lambda x: u(0,x)**2
                    vSquaredReferenceFunction = lambda x: v(0,x)**2
                    if itemporal==0:
                        plt.semilogy(xEval,np.sqrt(uSquaredErrorFunction(xEval)/uSquaredReferenceFunction(xEval)))
                        plt.legend()
                    #Compute Model Coeff
                    modelCoeff=np.empty((tEval.size,y0.size))
                    modelCoeff[0]=y0
                    for i in range(modelCoeff.shape[0]-1):
                        odeOut= scipy.integrate.solve_ivp(lambda t,y: model.dydtSource(y,t,sourceFunction),(tEval[i],tEval[i+1]),modelCoeff[i], method='BDF',atol=1e-13,rtol=1e-13)
                        modelCoeff[i+1] = odeOut.y[:,-1]
                        if odeOut.status!=0:
                            print("Warning: ode solver terminated prematurely")
                    #Check for error between model coeffecients at t=0 as outputed by ODE function and the true
                    y0error=np.max(modelCoeff[0,:]-y0)
                    if y0error>10**(-14):
                        print("Warning: odeint has non-zero error in y0")
                        print("y0 error: %03e" % (y0error,))
                        
                    #Evalute MMS and model at plot points
                    solutions[iColl,iElem,itemporal,iorder,0,0]=u(tEval,xEval)
                    solutions[iColl,iElem,itemporal,iorder,0,1]=model.eval(xEval,modelCoeff, output="u")
                    solutions[iColl,iElem,itemporal,iorder,1,0]=v(tEval,xEval)
                    solutions[iColl,iElem,itemporal,iorder,1,1]=model.eval(xEval,modelCoeff, output="v")

                    #Compute Error
                    uErrorFunction = lambda x: model.eval(x,modelCoeff,output="u") - u(tEval,x)
                    vErrorFunction = lambda x: model.eval(x,modelCoeff,output="v")- v(tEval,x)
                    uSquaredErrorFunction = lambda x: uErrorFunction(x)**2
                    vSquaredErrorFunction = lambda x: vErrorFunction(x)**2
                    uSquaredReferenceFunction = lambda x: u(tEval,x)**2
                    vSquaredReferenceFunction = lambda x: v(tEval,x)**2
                    # print(uSquaredErrorFunction(xEval).shape)
                    # print(uErrorFunction(xEval)[0,:])
                    # print(uSquaredReferenceFunction(xEval)[0,:])
                    quadOrder=spatialOrder**2
                    #quadOrder="auto"
                    #print("quadOrder: ", quadOrder)
                    uErrorL2,uErrorL2space = computeL2error(model,uSquaredErrorFunction,uSquaredReferenceFunction,tEval,order=quadOrder)
                    vErrorL2,vErrorL2space = computeL2error(model,vSquaredErrorFunction,vSquaredReferenceFunction,tEval,order=quadOrder)
                    uErrorLinf,uErrorLinfSpace = computeLinfError(uErrorFunction(xEval),u(tEval,xEval))
                    vErrorLinf,vErrorLinfSpace = computeLinfError(vErrorFunction(xEval),u(tEval,xEval))
                    #print(np.mean(np.sqrt(uSquaredErrorFunction(xEval)[0,:]/uSquaredReferenceFunction(xEval)[0,:])))
                    #print(uErrorL2space[0])
                    error[iColl,iElem,itemporal,iorder,0,0]=uErrorL2
                    error[iColl,iElem,itemporal,iorder,0,1]=uErrorLinf
                    error[iColl,iElem,itemporal,iorder,1,0]=vErrorL2
                    error[iColl,iElem,itemporal,iorder,1,1]=vErrorLinf
                    errorSpace[iColl,iElem,itemporal,iorder,0,0,:]=uErrorL2space
                    errorSpace[iColl,iElem,itemporal,iorder,0,1,:]=uErrorLinfSpace
                    errorSpace[iColl,iElem,itemporal,iorder,1,0,:]=vErrorL2space
                    errorSpace[iColl,iElem,itemporal,iorder,1,1,:]=vErrorLinfSpace

        spatialConvergenceRates[iColl]=computeConvergenceRates(1/np.array(nElems),errorSpace[iColl])
        jointConvergenceRates[iColl]=computeConvergenceRates(1/np.array(nElems),error[iColl])


    return error, solutions, jointConvergenceRates, errorSpace, spatialConvergenceRates


def computeConvergenceRates(discretizations,errors):
    #Check discretizations and errors have the same dimensions or that discretizations is 1D
    if discretizations.ndim==1:
        assert(discretizations.shape[0]==errors.shape[0])
    else:
        assert(discretizations.shape==errors.shape)
    #Define convergence rates as having the same shape as discretizations but with 1 less in the first dimension
    convergenceRates=np.log(errors[0:-1].T/errors[1:].T)/np.log(discretizations[0:-1]/discretizations[1:])
    return convergenceRates.T



def constructPolynomialMMSsolutionFunction(spatialOrder,params,temporal,temporaldt):
    uSpatialCoeff=-np.ones((spatialOrder+1,))
    vSpatialCoeff=-np.ones((spatialOrder+1,))
    
    uSpatialCoeff[1]=np.dot(np.arange(2,spatialOrder+1),np.ones((spatialOrder-1)))
    vSpatialCoeff[1]=np.dot(np.arange(2,spatialOrder+1),np.ones((spatialOrder-1)))
    uSpatialCoeff[0]=uSpatialCoeff[1]/params["PeM-boundary"]
    vSpatialCoeff[0]=(vSpatialCoeff[1]/params["PeT-boundary"]+params["f"]*(vSpatialCoeff[1]-(spatialOrder-1)))/(1-params["f"])
    
    dudxSpatialCoeff=uSpatialCoeff[1:]*np.arange(1,spatialOrder+1)
    dvdxSpatialCoeff=vSpatialCoeff[1:]*np.arange(1,spatialOrder+1)

    dudx2SpatialCoeff=dudxSpatialCoeff[1:]*np.arange(1,spatialOrder)
    dvdx2SpatialCoeff=dvdxSpatialCoeff[1:]*np.arange(1,spatialOrder)

    # print("uSpatialCoeff: ", uSpatialCoeff)
    # print("vSpatialCoeff: ", vSpatialCoeff)
    # print("dudxSpatialCoeff: ", dudxSpatialCoeff)
    # print("dudxSpatialCoeff: ", dvdxSpatialCoeff)
    # print("dudx2SpatialCoeff: ", dudx2SpatialCoeff)
    # print("dudx2SpatialCoeff: ", dvdx2SpatialCoeff)

    u = lambda t,x: np.outer(temporal(t),np.sum(np.power.outer(x,np.arange(0,spatialOrder+1))*uSpatialCoeff,axis=-1)).squeeze()
    v = lambda t,x: np.outer(temporal(t),np.sum(np.power.outer(x,np.arange(0,spatialOrder+1))*vSpatialCoeff,axis=-1)).squeeze()
    dudx = lambda t,x: np.outer(temporal(t),np.sum(np.power.outer(x,np.arange(0,spatialOrder))*dudxSpatialCoeff,axis=-1)).squeeze()
    dvdx = lambda t,x: np.outer(temporal(t),np.sum(np.power.outer(x,np.arange(0,spatialOrder))*dvdxSpatialCoeff,axis=-1)).squeeze()
    dudx2 = lambda t,x: np.outer(temporal(t),np.sum(np.power.outer(x,np.arange(0,spatialOrder-1))*dudx2SpatialCoeff,axis=-1)).squeeze()
    dvdx2 = lambda t,x: np.outer(temporal(t),np.sum(np.power.outer(x,np.arange(0,spatialOrder-1))*dvdx2SpatialCoeff,axis=-1)).squeeze()
    dudt = lambda t,x: np.outer(temporaldt(t),np.sum(np.power.outer(x,np.arange(0,spatialOrder+1))*uSpatialCoeff,axis=-1)).squeeze()
    dvdt = lambda t,x: np.outer(temporaldt(t),np.sum(np.power.outer(x,np.arange(0,spatialOrder+1))*vSpatialCoeff,axis=-1)).squeeze()
    t=np.array([0])
    x=np.array([0,1/2,1])

    return u, dudt, dudx, dudx2, v, dvdt, dvdx, dvdx2

def constructSinMMssolutionFunction(params,temporal,temporaldt):
    freq=np.pi 
    weight=1

    linearCoeff = -freq*weight*np.cos(freq)
    uConstCoeff = 1/params["PeM-boundary"]*(freq*weight+linearCoeff)
    vConstCoeff = ((freq*weight+linearCoeff)/params["PeT-boundary"]+params["f"]*(weight*np.sin(freq)+linearCoeff))/(1-params["f"])


    u= lambda t,x: np.outer(temporal(t),weight*np.sin(freq*x)+linearCoeff*x+uConstCoeff).squeeze()
    v= lambda t,x: np.outer(temporal(t),weight*np.sin(freq*x)+linearCoeff*x+vConstCoeff).squeeze()
    dudt= lambda t,x: np.outer(temporaldt(t),weight*np.sin(freq*x)+linearCoeff*x+uConstCoeff).squeeze()
    dvdt= lambda t,x: np.outer(temporaldt(t),weight*np.sin(freq*x)+linearCoeff*x+vConstCoeff).squeeze()
    dudx= lambda t,x: np.outer(temporal(t),freq*weight*np.cos(freq*x)+linearCoeff).squeeze()
    dvdx= lambda t,x: np.outer(temporal(t),freq*weight*np.cos(freq*x)+linearCoeff).squeeze()
    dudx2= lambda t,x: np.outer(temporal(t),-(freq**2)*weight*np.sin(freq*x)).squeeze()
    dvdx2= lambda t,x: np.outer(temporal(t),-(freq**2)*weight*np.sin(freq*x)).squeeze()

    return u, dudt, dudx, dudx2, v, dvdt, dvdx, dvdx2

def constructNonSeperableMMSsolutionFunction(params,spatialOrder):
    uCoeff = np.zeros((spatialOrder+1), dtype=object)
    vCoeff = np.zeros((spatialOrder+1), dtype=object)
    dudtCoeff = np.zeros((spatialOrder+1), dtype=object)
    dvdtCoeff = np.zeros((spatialOrder+1), dtype=object)
    # Fill higher-order coeffecients
    for n in range(2,spatialOrder+1):
        uCoeff[n] = lambda t: np.sin(2*n*np.pi*t)+np.cos(2*n*np.pi*t)
        dudtCoeff[n] = lambda t: 2*n*np.pi*(np.cos(2*n*np.pi*t)-np.sin(2*n*np.pi*t))
        vCoeff[n] = lambda t: np.sin(2*n*np.pi*t)+np.cos(2*n*np.pi*t)
        dvdtCoeff[n] = lambda t: 2*n*np.pi*(np.cos(2*n*np.pi*t)-np.sin(2*n*np.pi*t))

    # Fill first order coeffecients from right- boundary condition
    uCoeff[1] = lambda t: -np.sum([n*uCoeff[n](t) for n in range(2,spatialOrder+1)], axis=0)
    vCoeff[1] = lambda t: -np.sum([n*vCoeff[n](t) for n in range(2,spatialOrder+1)], axis=0)


    uCoeff[0] = lambda t: uCoeff[1](t)/params["PeM-boundary"]
    vCoeff[0] = lambda t: (vCoeff[1](t)/params["PeT-boundary"]+params["f"]*np.sum([vCoeff[i](t) for i in range(1,spatialOrder+1)],axis=0)) / (1-params["f"])


    u = lambda t,x: np.sum([uCoeff[n](t)*x**n for n in range(0,spatialOrder+1)],axis=0)
    v = lambda t,x: np.sum([vCoeff[n](t)*x**n for n in range(0,spatialOrder+1)],axis=0)
    dudx = lambda t,x: np.sum([n*uCoeff[n](t)*x**(n-1) for n in range(1,spatialOrder+1)],axis=0)
    dvdx = lambda t,x: np.sum([n*vCoeff[n](t)*x**(n-1) for n in range(1,spatialOrder+1)],axis=0)
    dudx2 = lambda t,x: np.sum([n*(n-1)*uCoeff[n](t)*x**(n-2) for n in range(2,spatialOrder+1)],axis=0)
    dvdx2 = lambda t,x: np.sum([n*(n-1)*vCoeff[n](t)*x**(n-2) for n in range(2,spatialOrder+1)],axis=0)

    dudt = lambda t,x: np.sum([dudtCoeff[n](t)*x**n for n in range(0,spatialOrder+1)],axis=0)
    dvdt = lambda t,x: np.sum([dvdtCoeff[n](t)*x**n for n in range(0,spatialOrder+1)],axis=0)

    return u, dudt, dudx, dudx2, v, dvdt, dvdx, dvdx2



def constructSourceTermFunction(u, dudt, dudx, dudx2, v, dvdt, dvdx, dvdx2,params):
    #Construct Source term
    sourceU = lambda t:  dudt(t)+dudx(t)-(dudx2(t)/params["PeM"]+params["Da"]*(1-u(t))*np.exp(params["gamma"]*params["beta"]*v(t)/(1+params["beta"]*v(t))))
    sourceV = lambda t:  params["Le"]*dvdt(t)+dvdx(t)-(dvdx2(t)/params["PeT"]+params["Da"]*(1-u(t))*np.exp(params["gamma"]*params["beta"]*v(t)/(1+params["beta"]*v(t)))+params["delta"]*(params["vH"]-v(t)))

    return lambda t: np.concatenate((sourceU(t),sourceV(t)),axis=-1)


    


#!!!!!!!!!Change all of these to be (time,space) for consistency elsewhere in code
def computeL2error(model,squaredErrorFunction,squaredReferenceFunction,tPoints,order="auto"):
    errorL2Squared, errorL2SpaceSquared = model.integrate(squaredErrorFunction,tPoints,integrateTime=True,order=order)
    #print("Pre-relatives L2 error: ", errorL2SpaceSquared)
    referenceL2Squared,referenceL2SpaceSquared=model.integrate(squaredReferenceFunction,tPoints,integrateTime=True,order=order)
    errorL2=np.sqrt(errorL2Squared/referenceL2Squared)
    errorL2Space=np.sqrt(errorL2SpaceSquared/referenceL2SpaceSquared)
    return errorL2, errorL2Space

def computeLinfError(error,reference):
    errorLinfSpace=np.max(np.abs(error),axis=1)/np.max(np.abs(reference),axis=1)
    errorLinf=np.max(np.abs(error))/np.max(np.abs(reference))
    return errorLinf, errorLinfSpace
