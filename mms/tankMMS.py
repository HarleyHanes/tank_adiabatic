import numpy as np
import scipy
from tankModel.TankModel import TankModel


def runMMStest(spatialSolOrders,nCollocations,nElems,xEval,tEval,params,verbosity = 0):
    temporals=[lambda t: 1+0*t,lambda t: t]
    temporalsdt=[lambda t: 0*t,lambda t: 1+0*t]
    #temporals = [lambda t: 1+0*t, lambda t: t, lambda t: t**2, lambda t: np.sin(t)]
    #temporalsdt = [lambda t: 0*t, lambda t: 1+0*t, lambda t: 2*t, lambda t: np.cos(t)]
    error= np.empty((len(nCollocations),len(nElems),len(temporals),len(spatialSolOrders),2,2))
    solutions= np.empty((len(nCollocations),len(nElems),len(temporals),len(spatialSolOrders),2,2,tEval.size,xEval.size))
    convergenceRates = np.empty((len(nCollocations),len(nElems)-1,len(temporals),len(spatialSolOrders),2,2))
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
                    modelCoeff = scipy.integrate.odeint(lambda y,t: model.dydtSource(y,t,sourceFunction),y0,tEval)
                    #Evalue manufactured solution at integration points
                    u, dudt, dudx, dudx2, v, dvdt, dvdx, dvdx2 = constructMMSsolutionFunction(xEval,spatialOrder,params,temporals[itemporal],temporalsdt[itemporal])
                    uMMSsol=u(tEval)
                    vMMSsol=v(tEval)
                    #Evaluate model at integration points
                    uModelSol, vModelSol = model.eval(xEval,modelCoeff, seperated=True)
                    #Compute Error
                    uErrorL2, uErrorLinf = computeMMSerror(uModelSol,uMMSsol,tEval[-1],tEval[0])
                    vErrorL2, vErrorLinf = computeMMSerror(vModelSol,vMMSsol,tEval[-1],tEval[0])
                    #Save Error
                    error[iColl,iElem,itemporal,iorder,0,0]=uErrorL2
                    error[iColl,iElem,itemporal,iorder,0,1]=uErrorLinf
                    error[iColl,iElem,itemporal,iorder,1,0]=vErrorL2
                    error[iColl,iElem,itemporal,iorder,1,1]=vErrorLinf

                    solutions[iColl,iElem,itemporal,iorder,0,0]=uMMSsol
                    solutions[iColl,iElem,itemporal,iorder,0,1]=uModelSol
                    solutions[iColl,iElem,itemporal,iorder,1,0]=vMMSsol
                    solutions[iColl,iElem,itemporal,iorder,0,1]=uModelSol
            if iElem!=0:
                convergenceRates[iColl,iElem-1,:,:,:,:]=(error[iColl,iElem-1,:,:,:,:]/error[iColl,iElem,:,:,:,:])*(nElems[iElem-1]/nElems[iElem])


    return error, solutions, convergenceRates




def constructMMSsolutionFunction(x,spatialOrder,params,temporal,temporaldt):

    #Construct sums of monomials of spatialOrder 2+ (assuming coeffecients of -1 for now)
    monomialSum = np.sum(x**(np.arange(2,spatialOrder+1)*np.ones(x.shape+(spatialOrder-1,))).transpose(),axis=0)
    dmonomialSumdx = np.sum(((x**((np.arange(1,spatialOrder)*np.ones(x.shape+(spatialOrder-1,)))).transpose()).transpose()*np.arange(2,spatialOrder+1)).transpose(),axis=0)
    dmonomialSumdx2 = np.sum(((x**((np.arange(spatialOrder-1)*np.ones(x.shape+(spatialOrder-1,)))).transpose()).transpose()*np.arange(2,spatialOrder+1)*np.arange(1,spatialOrder)).transpose(),axis=0)
                
    #Apply corrects to first spatialOrder coeffecients so that solutions satisfy BC
    linearCoeff=-np.dot(np.arange(2,spatialOrder+1),np.ones((spatialOrder-1)))
    uConstantCoeff=linearCoeff/params["PeM"]
    vConstantCoeff=(params["f"]*(spatialOrder-1)+linearCoeff*(params["f"]+1/params["PeT"]))/(1-params["f"])

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
    sourceU = lambda t:  dudt(t)-(dudx2(t)+params["PeM"]*dudx(t)+params["Da"]*(1-u(t))*np.exp(params["gamma"]*params["beta"]*v(t)/(1+params["beta"]*v(t))))
    sourceV = lambda t:  params["Le"]*dvdt(t)-(dvdx2(t)+params["PeT"]*dvdx(t)+params["Da"]*(1-u(t))*np.exp(params["gamma"]*params["beta"]*v(t)/(1+params["beta"]*v(t)))+params["delta"]*(params["vH"]-v(t)))

    return lambda t: np.concatenate((sourceU(t),sourceV(t)),axis=-1)


def computeMMSerror(computedSolution,mmsSolution,tMax,tMin,rule="trapezoid"):
    if rule=="trapezoid":
        difference = (mmsSolution-computedSolution)
        errorL2space = np.sqrt((np.sum(2*(difference**2),axis=1)-difference[:,0]**2-difference[:,-1]**2)/2)
        errorL2=np.sqrt((np.sum(2*(errorL2space**2))-errorL2space[0]**2-errorL2space[-1]**2)/2*(tMax-tMin))
        mmsNormSpace = np.sqrt((np.sum(2*(mmsSolution**2),axis=1)-mmsSolution[:,0]**2-mmsSolution[:,-1]**2)/2)
        mmsNorm=np.sqrt((np.sum(2*(mmsNormSpace**2))-mmsNormSpace[0]**2-mmsNormSpace[-1]**2)/2*(tMax-tMin))
    errorLinf=np.max(np.abs(difference))/np.max(np.abs(mmsSolution))
    return errorL2/mmsNorm, errorLinf
