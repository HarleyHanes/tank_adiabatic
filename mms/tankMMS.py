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
                    u, dudt, dudx, dudx2, v, dvdt, dvdx, dvdx2 = constructMMSsolutionFunction(x,spatialOrder,params,temporals[itemporal],temporalsdt[itemporal])
                    sourceFunction = constructSourceTermFunction(u, dudt, dudx, dudx2, v, dvdt, dvdx, dvdx2,params)
                    y0=np.append(u(tEval[0]),v(tEval[0]))
                    #Compute Model Coeff
                    modelCoeff=np.empty((tEval.size,y0.size))
                    modelCoeff[0]=y0
                    for i in range(modelCoeff.shape[0]-1):
                        odeOut= scipy.integrate.solve_ivp(lambda t,y: model.dydtSource(y,t,sourceFunction),(tEval[i],tEval[i+1]),modelCoeff[i], method='BDF',atol=1e-14,rtol=1e-14)
                        modelCoeff[i+1] = odeOut.y[:,-1]
                        if odeOut.status!=0:
                            print("Warning: ode solver terminated prematurely")
                    #Check for error between model coeffecients at t=0 as outputed by ODE function and the true
                    y0error=np.sqrt(np.sum((modelCoeff[0,:]-y0)**2))
                    if y0error>10**(-14):
                        print("Warning: odeint has non-zero error in y0")
                        print("y0 error: %03e" % (y0error,))
                    #Evalue manufactured solution at integration points
                    u, dudt, dudx, dudx2, v, dvdt, dvdx, dvdx2 = constructMMSsolutionFunction(xEval,spatialOrder,params,temporals[itemporal],temporalsdt[itemporal])
                    uMMSsol=u(tEval)
                    vMMSsol=v(tEval)
                    #Evaluate model at integration points
                    uModelSol, vModelSol = model.eval(xEval,modelCoeff, seperated=True)
                    #Confirm that for exact spatial cases, the error at first step is near-zero
                    # if spatialOrder<=(nCollocations[iColl]+1):
                    #     uInitialError = np.max(np.abs((uMMSsol[0]-uModelSol[0])/uMMSsol[0]))
                    #     vInitialError = np.max(np.abs((vMMSsol[0]-vModelSol[0])/vMMSsol[0]))
                    #     if uInitialError>1e-13 or vInitialError>1e-13:
                    #         print("Warning: error at start of solve for exact problem")
                    #         print("u Linf error at t=0: ", uInitialError)
                    #         print("v Linf error at t=0: ", vInitialError)

                    #Compute Error
                    uErrorL2, uErrorLinf, uErrorL2space, uErrorLinfSpace = computeMMSerror(uModelSol,uMMSsol,tEval[-1],tEval[0])
                    vErrorL2, vErrorLinf, vErrorL2space, vErrorLinfSpace = computeMMSerror(vModelSol,vMMSsol,tEval[-1],tEval[0])
                    #Save Error
                    error[iColl,iElem,itemporal,iorder,0,0]=uErrorL2
                    error[iColl,iElem,itemporal,iorder,0,1]=uErrorLinf
                    error[iColl,iElem,itemporal,iorder,1,0]=vErrorL2
                    error[iColl,iElem,itemporal,iorder,1,1]=vErrorLinf
                    errorSpace[iColl,iElem,itemporal,iorder,0,0,:]=uErrorL2space
                    errorSpace[iColl,iElem,itemporal,iorder,0,1,:]=uErrorLinfSpace
                    errorSpace[iColl,iElem,itemporal,iorder,1,0,:]=vErrorL2space
                    errorSpace[iColl,iElem,itemporal,iorder,1,1,:]=vErrorLinfSpace

                    solutions[iColl,iElem,itemporal,iorder,0,0]=uMMSsol
                    solutions[iColl,iElem,itemporal,iorder,0,1]=uModelSol
                    solutions[iColl,iElem,itemporal,iorder,1,0]=vMMSsol
                    solutions[iColl,iElem,itemporal,iorder,1,1]=vModelSol
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


def constructMMSsolutionFunction(x,spatialOrder,params,temporal,temporaldt):

    #Construct sums of monomials of spatialOrder 2+ (assuming coeffecients of -1 for now)
    monomialSum = np.sum(-x**(np.arange(2,spatialOrder+1)*np.ones(x.shape+(spatialOrder-1,))).transpose(),axis=0)
    dmonomialSumdx = np.sum(((-x**((np.arange(1,spatialOrder)*np.ones(x.shape+(spatialOrder-1,)))).transpose()).transpose()*np.arange(2,spatialOrder+1)).transpose(),axis=0)
    dmonomialSumdx2 = np.sum((-(x**((np.arange(spatialOrder-1)*np.ones(x.shape+(spatialOrder-1,)))).transpose()).transpose()*np.arange(2,spatialOrder+1)*np.arange(1,spatialOrder)).transpose(),axis=0)
                
    #Apply corrections to linear spatialOrder coeffecients so that solutions satisfy BC
    linearCoeff=np.dot(np.arange(2,spatialOrder+1),np.ones((spatialOrder-1)))  #Note that - sign in sum and -1 coeffecients on monomial terms cancel
    uConstantCoeff=linearCoeff/params["PeM"]
    vConstantCoeff=(linearCoeff/params["PeT"]+params["f"]*(linearCoeff-(spatialOrder-1)))/(1-params["f"])
 
    uSpatialComponent = monomialSum + uConstantCoeff + linearCoeff*x
    vSpatialComponent = monomialSum + vConstantCoeff + linearCoeff*x
    dudxSpatialComponent = dmonomialSumdx+ linearCoeff
    dvdxSpatialComponent = dmonomialSumdx+ linearCoeff
                
    dudx2SpatialComponent = dmonomialSumdx2
    dvdx2SpatialComponent = dmonomialSumdx2

    u = lambda t: np.outer(temporal(t),uSpatialComponent).squeeze()
    dudt  = lambda t: np.outer(temporaldt(t),uSpatialComponent).squeeze()
    dudx  = lambda t: np.outer(temporal(t),dudxSpatialComponent).squeeze()
    dudx2 = lambda t: np.outer(temporal(t),dudx2SpatialComponent).squeeze()
    v=lambda t,: np.outer(temporal(t),vSpatialComponent).squeeze()
    dvdt  = lambda t: np.outer(temporaldt(t),vSpatialComponent).squeeze()
    dvdx  = lambda t: np.outer(temporal(t),dvdxSpatialComponent).squeeze()
    dvdx2 = lambda t: np.outer(temporal(t),dvdx2SpatialComponent).squeeze()

    return u, dudt, dudx, dudx2, v, dvdt, dvdx, dvdx2

def constructSourceTermFunction(u, dudt, dudx, dudx2, v, dvdt, dvdx, dvdx2,params):
    #Construct Source term
    sourceU = lambda t:  dudt(t)+dudx(t)-(dudx2(t)/params["PeM"]+params["Da"]*(1-u(t))*np.exp(params["gamma"]*params["beta"]*v(t)/(1+params["beta"]*v(t))))
    sourceV = lambda t:  params["Le"]*dvdt(t)+dvdx(t)-(dvdx2(t)/params["PeT"]+params["Da"]*(1-u(t))*np.exp(params["gamma"]*params["beta"]*v(t)/(1+params["beta"]*v(t)))+params["delta"]*(params["vH"]-v(t)))

    return lambda t: np.concatenate((sourceU(t),sourceV(t)),axis=-1)


    

def computeMMSerror(computedSolution,mmsSolution,tMax,tMin,rule="trapezoid"):
    if rule=="trapezoid":
        difference = (mmsSolution-computedSolution)
        errorL2space = np.sqrt((np.sum(2*(difference**2),axis=1)-difference[:,0]**2-difference[:,-1]**2)/2)
        errorL2=np.sqrt((np.sum(2*(errorL2space**2))-errorL2space[0]**2-errorL2space[-1]**2)/2*(tMax-tMin))
        mmsNormSpace = np.sqrt((np.sum(2*(mmsSolution**2),axis=1)-mmsSolution[:,0]**2-mmsSolution[:,-1]**2)/2)
        mmsNorm=np.sqrt((np.sum(2*(mmsNormSpace**2))-mmsNormSpace[0]**2-mmsNormSpace[-1]**2)/2*(tMax-tMin))
    errorLinfSpace=np.max(np.abs(difference),axis=1)/np.max(np.abs(mmsSolution),axis=1)
    errorLinf=np.max(np.abs(difference))/np.max(np.abs(mmsSolution))
    return errorL2/mmsNorm, errorLinf, errorL2space/mmsNormSpace, errorLinfSpace
