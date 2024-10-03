import numpy as np
import scipy
from tankModel.TankModel import TankModel


def runMMStest(spatialSolOrders,nCollocations,nElems,xEval,tEval,params,verbosity = 0):
    #Parse inputs
    temporals=[lambda t: 1+0*t,lambda t: 1+1*t, lambda t: 1+t*np.exp(2*t)]
    temporalsdt=[lambda t: 0*t,lambda t: 1+0*t, lambda t: np.exp(2*t)+2*t*np.exp(2*t)]
    #temporals = [lambda t: 1+0*t, lambda t: t, lambda t: t**2, lambda t: np.sin(t)]
    #temporalsdt = [lambda t: 0*t, lambda t: 1+0*t, lambda t: 2*t, lambda t: np.cos(t)]
    error = np.empty((len(nCollocations),len(nElems),len(temporals),len(spatialSolOrders),2,2))
    errorSpace = np.empty((len(nCollocations),len(nElems),len(temporals),len(spatialSolOrders),2,2,tEval.size))
    #Pre allocate solutions, the (2,2) indices are for (u,v) and then (MMS,Model) where MMS is true value and Model is computed value
    solutions= np.empty((len(nCollocations),len(nElems),len(temporals),len(spatialSolOrders),2,2,tEval.size,xEval.size))
    jointConvergenceRates = np.empty((len(nCollocations),len(nElems)-1,len(temporals),len(spatialSolOrders),2,2))
    spatialConvergenceRates = np.empty((len(nCollocations),len(nElems)-1,len(temporals),len(spatialSolOrders),2,2,tEval.size))
    for iColl in range(len(nCollocations)):
        for iElem in range(len(nElems)):
            if verbosity > 0 :
                print("Testing for (collocation points, elements): (" + str(nCollocations[iColl])+ ", " + str(nElems[iElem])+ ")")
            # Generate tankModel
            model = TankModel(nCollocation=nCollocations[iColl],nElements=nElems[iElem],spacing="legendre",bounds=[0,1],params=params, verbosity =verbosity)
            # Set discretization
            x = model.collocationPoints
            #Loop through exact monomial cases
            #Loop through temporal functions
            for itemporal in range(len(temporals)):
                if verbosity > 0:
                    print("iTemporal: ", itemporal)
                for iorder in range(len(spatialSolOrders)):
                    spatialOrder=spatialSolOrders[iorder]
                    if verbosity > 0:
                        print("Order: ", spatialOrder)
                    #Construct solutions that are sum of monomials up to spatialOrder 
                    if spatialOrder < 2:
                        raise Exception("Error, spatialOrder less than 2 entered. Only 2+ spatialOrder polynomials can satisfy BC")
                    u, dudt, dudx, dudx2, v, dvdt, dvdx, dvdx2 = constructMMSsolutionFunction(spatialOrder,params,temporals[itemporal],temporalsdt[itemporal])
                    sourceFunction = constructSourceTermFunction(lambda t: u(t,model.collocationPoints), 
                                                                 lambda t: dudt(t,model.collocationPoints), 
                                                                 lambda t: dudx(t,model.collocationPoints), 
                                                                 lambda t: dudx2(t,model.collocationPoints), 
                                                                 lambda t: v(t,model.collocationPoints), 
                                                                 lambda t: dvdt(t,model.collocationPoints), 
                                                                 lambda t: dvdx(t,model.collocationPoints), 
                                                                 lambda t: dvdx2(t,model.collocationPoints),params)
                    
                    y0=np.append(u(tEval[0],model.collocationPoints),v(tEval[0],model.collocationPoints))
                    #Compute Model Coeff
                    modelCoeff=np.empty((tEval.size,y0.size))
                    modelCoeff[0]=y0
                    for i in range(modelCoeff.shape[0]-1):
                        odeOut= scipy.integrate.solve_ivp(lambda t,y: model.dydtSource(y,t,sourceFunction),(tEval[i],tEval[i+1]),modelCoeff[i], method='BDF',atol=1e-13,rtol=1e-13)
                        modelCoeff[i+1] = odeOut.y[:,-1]
                        if odeOut.status!=0:
                            print("Warning: ode solver terminated prematurely")
                    #Check for error between model coeffecients at t=0 as outputed by ODE function and the true
                    y0error=np.sqrt(np.sum((modelCoeff[0,:]-y0)**2))
                    if y0error>10**(-14):
                        print("Warning: odeint has non-zero error in y0")
                        print("y0 error: %03e" % (y0error,))
                        
                    #Evalute MMS and model at plot points
                    uMMSsol=u(tEval,xEval)
                    vMMSsol=v(tEval,xEval)
                    uModelSol, vModelSol = model.eval(xEval,modelCoeff, output="seperated")
                    solutions[iColl,iElem,itemporal,iorder,0,0]=uMMSsol
                    solutions[iColl,iElem,itemporal,iorder,0,1]=uModelSol
                    solutions[iColl,iElem,itemporal,iorder,1,0]=vMMSsol
                    solutions[iColl,iElem,itemporal,iorder,1,1]=vModelSol


                    #Compute Error
                    uErrorFunction = lambda x: model.eval(x,modelCoeff,output="u") - u(tEval,x)
                    vErrorFunction = lambda x: model.eval(x,modelCoeff,output="v")- v(tEval,x)
                    uSquaredErrorFunction = lambda x: uErrorFunction(x)**2
                    vSquaredErrorFunction = lambda x: vErrorFunction(x)**2
                    uSquaredReferenceFunction = lambda x: u(tEval,x)**2
                    vSquaredReferenceFunction = lambda x: v(tEval,x)**2

                    uErrorL2,uErrorL2space = computeL2error(model,uSquaredErrorFunction,uSquaredReferenceFunction,tEval)
                    vErrorL2,vErrorL2space = computeL2error(model,vSquaredErrorFunction,vSquaredReferenceFunction,tEval)
                    uErrorLinf,uErrorLinfSpace = computeLinfError(uErrorFunction(xEval),u(tEval,xEval))
                    vErrorLinf,vErrorLinfSpace = computeLinfError(vErrorFunction(xEval),u(tEval,xEval))

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



def constructMMSsolutionFunction(spatialOrder,params,temporal,temporaldt):

    uSpatialCoeff=-np.ones((spatialOrder+1,))
    vSpatialCoeff=-np.ones((spatialOrder+1,))

    uSpatialCoeff[1]=np.dot(np.arange(2,spatialOrder+1),np.ones((spatialOrder-1)))
    vSpatialCoeff[1]=np.dot(np.arange(2,spatialOrder+1),np.ones((spatialOrder-1)))
    uSpatialCoeff[0]=uSpatialCoeff[1]/params["PeM"]
    vSpatialCoeff[0]=(vSpatialCoeff[1]/params["PeT"]+params["f"]*(vSpatialCoeff[1]-(spatialOrder-1)))/(1-params["f"])
    
    dudxSpatialCoeff=uSpatialCoeff[1:]*np.arange(1,spatialOrder+1)
    dvdxSpatialCoeff=vSpatialCoeff[1:]*np.arange(1,spatialOrder+1)

    dudx2SpatialCoeff=dudxSpatialCoeff[1:]*np.arange(1,spatialOrder)
    dvdx2SpatialCoeff=dvdxSpatialCoeff[1:]*np.arange(1,spatialOrder)

    u = lambda t,x: np.outer(temporal(t),np.sum(np.power.outer(x,np.arange(0,spatialOrder+1))*uSpatialCoeff,axis=-1)).squeeze()
    v = lambda t,x: np.outer(temporal(t),np.sum(np.power.outer(x,np.arange(0,spatialOrder+1))*vSpatialCoeff,axis=-1)).squeeze()
    dudx = lambda t,x: np.outer(temporal(t),np.sum(np.power.outer(x,np.arange(0,spatialOrder))*dudxSpatialCoeff,axis=-1)).squeeze()
    dvdx = lambda t,x: np.outer(temporal(t),np.sum(np.power.outer(x,np.arange(0,spatialOrder))*dvdxSpatialCoeff,axis=-1)).squeeze()
    dudx2 = lambda t,x: np.outer(temporal(t),np.sum(np.power.outer(x,np.arange(0,spatialOrder-1))*dudx2SpatialCoeff,axis=-1)).squeeze()
    dvdx2 = lambda t,x: np.outer(temporal(t),np.sum(np.power.outer(x,np.arange(0,spatialOrder-1))*dvdx2SpatialCoeff,axis=-1)).squeeze()
    dudt = lambda t,x: np.outer(temporaldt(t),np.sum(np.power.outer(x,np.arange(0,spatialOrder+1))*uSpatialCoeff,axis=-1)).squeeze()
    dvdt = lambda t,x: np.outer(temporaldt(t),np.sum(np.power.outer(x,np.arange(0,spatialOrder+1))*vSpatialCoeff,axis=-1)).squeeze()


    return u, dudt, dudx, dudx2, v, dvdt, dvdx, dvdx2

def constructSourceTermFunction(u, dudt, dudx, dudx2, v, dvdt, dvdx, dvdx2,params):
    #Construct Source term
    sourceU = lambda t:  dudt(t)+dudx(t)-(dudx2(t)/params["PeM"]+params["Da"]*(1-u(t))*np.exp(params["gamma"]*params["beta"]*v(t)/(1+params["beta"]*v(t))))
    sourceV = lambda t:  params["Le"]*dvdt(t)+dvdx(t)-(dvdx2(t)/params["PeT"]+params["Da"]*(1-u(t))*np.exp(params["gamma"]*params["beta"]*v(t)/(1+params["beta"]*v(t)))+params["delta"]*(params["vH"]-v(t)))

    return lambda t: np.concatenate((sourceU(t),sourceV(t)),axis=-1)


    


#!!!!!!!!!Change all of these to be (time,space) for consistency elsewhere in code
def computeL2error(model,squaredErrorFunction,squaredReferenceFunction,tPoints):
    errorL2Squared, errorL2SpaceSquared = model.integrate(squaredErrorFunction,tPoints,integrateTime=True)
    referenceL2Squared,referenceL2SpaceSquared=model.integrate(squaredReferenceFunction,tPoints,integrateTime=True)
    errorL2=np.sqrt(errorL2Squared/referenceL2Squared)
    errorL2Space=np.sqrt(errorL2SpaceSquared/referenceL2SpaceSquared)
    return errorL2, errorL2Space

def computeLinfError(error,reference):
    errorLinfSpace=np.max(np.abs(error),axis=1)/np.max(np.abs(reference),axis=1)
    errorLinf=np.max(np.abs(error))/np.max(np.abs(reference))
    return errorLinf, errorLinfSpace
