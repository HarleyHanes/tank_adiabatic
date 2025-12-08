class RomData:
    def __init__(
        self,
        x,
        W,
        uTimeModes,
        uMean,
        uModes,
        uModesx,
        uModesxx,
        uModesWeighted,
        uModesInt,
        uRomMassMean,
        uRomFirstOrderMat,
        uRomFirstOrderMean,
        uRomSecondOrderMat,
        uRomSecondOrderMean,
        vTimeModes,
        vMean,
        vModes,
        vModesx,
        vModesxx,
        vModesWeighted,
        vModesInt,
        vRomMassMean,
        vRomFirstOrderMat,
        vRomFirstOrderMean,
        vRomSecondOrderMat,
        vRomSecondOrderMean,
        uSingularValues,
        uFullSpectra,
        vSingularValues,
        vFullSpectra,
        uNonlinDim,
        vNonlinDim,
        deimProjection,
        uNonLinProjection,
        vNonLinProjection,
    ):

        self.x = x
        self.nPoints = len(x)
        self.uNmodes = uModes.shape[1]
        self.vNmodes = vModes.shape[1]

        # Initialize all other attributes
        self.W = W
        self.uTimeModes = uTimeModes
        self.uMean = uMean
        self.uModes = uModes
        self.uModesx = uModesx
        self.uModesxx = uModesxx
        self.uModesWeighted = uModesWeighted
        self.uModesInt = uModesInt
        self.uRomMassMean = uRomMassMean
        self.uRomFirstOrderMat = uRomFirstOrderMat
        self.uRomFirstOrderMean = uRomFirstOrderMean
        self.uRomSecondOrderMat = uRomSecondOrderMat
        self.uRomSecondOrderMean = uRomSecondOrderMean
        self.vTimeModes = vTimeModes
        self.vMean = vMean
        self.vModes = vModes
        self.vModesx = vModesx
        self.vModesxx = vModesxx
        self.vModesWeighted = vModesWeighted
        self.vModesInt = vModesInt
        self.vRomMassMean = vRomMassMean
        self.vRomFirstOrderMat = vRomFirstOrderMat
        self.vRomFirstOrderMean = vRomFirstOrderMean
        self.vRomSecondOrderMat = vRomSecondOrderMat
        self.vRomSecondOrderMean = vRomSecondOrderMean
        self.uSingularValues = uSingularValues
        self.uFullSpectra = uFullSpectra
        self.vSingularValues = vSingularValues
        self.vFullSpectra = vFullSpectra
        self.uNonlinDim = uNonlinDim
        self.vNonlinDim = vNonlinDim
        self.uNonLinProjection = uNonLinProjection
        self.vNonLinProjection = vNonLinProjection
        self.deimProjection = deimProjection

    # Getter and setter methods with validation
    @property
    def nPoints(self):
        return self._nPoints

    @nPoints.setter
    def nPoints(self, value):
        if not isinstance(value, int):
            raise ValueError("nPoints must be an integer")
        self._nPoints = value

    @property
    def uNmodes(self):
        return self._uNmodes

    @uNmodes.setter
    def uNmodes(self, value):
        if not isinstance(value, int):
            raise ValueError("uNmodes must be an integer")
        self._uNmodes = value

    @property
    def vNmodes(self):
        return self._vNmodes

    @vNmodes.setter
    def vNmodes(self, value):
        if not isinstance(value, int):
            raise ValueError("vNmodes must be an integer")
        self._vNmodes = value

    def _validate_shape(self, array, shape, name):
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, but has shape {array.shape}")

    @property
    def W(self):
        return self._W

    @W.setter
    def W(self, value):
        self._validate_shape(value, (self.nPoints, self.nPoints), "W")
        self._W = value

    @property
    def uTimeModes(self):
        return self._uTimeModes

    @uTimeModes.setter
    def uTimeModes(self, value):
        self._validate_shape(value, (value.shape[0], self.uNmodes), "uTimeModes")
        self._uTimeModes = value

    @property
    def uMean(self):
        return self._uMean

    @uMean.setter
    def uMean(self, value):
        self._validate_shape(value, (self.nPoints,), "uMean")
        self._uMean = value

    @property
    def uModes(self):
        return self._uModes

    @uModes.setter
    def uModes(self, value):
        self._validate_shape(value, (self.nPoints, self.uNmodes), "uModes")
        self._uModes = value

    @property
    def uModesx(self):
        return self._uModesx

    @uModesx.setter
    def uModesx(self, value):
        self._validate_shape(value, (self.nPoints, self.uNmodes), "uModesx")
        self._uModesx = value

    @property
    def uModesxx(self):
        return self._uModesxx

    @uModesxx.setter
    def uModesxx(self, value):
        self._validate_shape(value, (self.nPoints, self.uNmodes), "uModesxx")
        self._uModesxx = value

    @property
    def uModesWeighted(self):
        return self._uModesWeighted

    @uModesWeighted.setter
    def uModesWeighted(self, value):
        self._validate_shape(value, (self.nPoints, self.uNmodes), "uModesWeighted")
        self._uModesWeighted = value

    @property
    def uModesInt(self):
        return self._uModesInt

    @uModesInt.setter
    def uModesInt(self, value):
        self._validate_shape(value, (self.uNmodes,), "uModesInt")
        self._uModesInt = value

    @property
    def uRomMassMean(self):
        return self._uRomMassMean

    @uRomMassMean.setter
    def uRomMassMean(self, value):
        self._validate_shape(value, (self.uNmodes,), "uRomMassMean")
        self._uRomMassMean = value

    @property
    def uRomFirstOrderMat(self):
        return self._uRomFirstOrderMat

    @uRomFirstOrderMat.setter
    def uRomFirstOrderMat(self, value):
        self._validate_shape(value, (self.uNmodes, self.uNmodes), "uRomFirstOrderMat")
        self._uRomFirstOrderMat = value

    @property
    def uRomFirstOrderMean(self):
        return self._uRomFirstOrderMean

    @uRomFirstOrderMean.setter
    def uRomFirstOrderMean(self, value):
        self._validate_shape(value, (self.uNmodes,), "uRomFirstOrderMean")
        self._uRomFirstOrderMean = value

    @property
    def uRomSecondOrderMat(self):
        return self._uRomSecondOrderMat

    @uRomSecondOrderMat.setter
    def uRomSecondOrderMat(self, value):
        self._validate_shape(value, (self.uNmodes, self.uNmodes), "uRomSecondOrderMat")
        self._uRomSecondOrderMat = value

    @property
    def uRomSecondOrderMean(self):
        return self._uRomSecondOrderMean

    @uRomSecondOrderMean.setter
    def uRomSecondOrderMean(self, value):
        self._validate_shape(value, (self.uNmodes,), "uRomSecondOrderMean")
        self._uRomSecondOrderMean = value

    @property
    def vTimeModes(self):
        return self._vTimeModes

    @vTimeModes.setter
    def vTimeModes(self, value):
        self._validate_shape(value, (value.shape[0], self.vNmodes), "vTimeModes")
        self._vTimeModes = value

    @property
    def vMean(self):
        return self._vMean

    @vMean.setter
    def vMean(self, value):
        self._validate_shape(value, (self.nPoints,), "vMean")
        self._vMean = value

    @property
    def vModes(self):
        return self._vModes

    @vModes.setter
    def vModes(self, value):
        self._validate_shape(value, (self.nPoints, self.vNmodes), "vModes")
        self._vModes = value

    @property
    def vModesx(self):
        return self._vModesx

    @vModesx.setter
    def vModesx(self, value):
        self._validate_shape(value, (self.nPoints, self.vNmodes), "vModesx")
        self._vModesx = value

    @property
    def vModesxx(self):
        return self._vModesxx

    @vModesxx.setter
    def vModesxx(self, value):
        self._validate_shape(value, (self.nPoints, self.vNmodes), "vModesxx")
        self._vModesxx = value

    @property
    def vModesWeighted(self):
        return self._vModesWeighted

    @vModesWeighted.setter
    def vModesWeighted(self, value):
        self._validate_shape(value, (self.nPoints, self.vNmodes), "vModesWeighted")
        self._vModesWeighted = value

    @property
    def vModesInt(self):
        return self._vModesInt

    @vModesInt.setter
    def vModesInt(self, value):
        self._validate_shape(value, (self.vNmodes,), "vModesInt")
        self._vModesInt = value

    @property
    def vRomMassMean(self):
        return self._vRomMassMean

    @vRomMassMean.setter
    def vRomMassMean(self, value):
        self._validate_shape(value, (self.vNmodes,), "vRomMassMean")
        self._vRomMassMean = value

    @property
    def vRomFirstOrderMat(self):
        return self._vRomFirstOrderMat

    @vRomFirstOrderMat.setter
    def vRomFirstOrderMat(self, value):
        self._validate_shape(value, (self.vNmodes, self.vNmodes), "vRomFirstOrderMat")
        self._vRomFirstOrderMat = value

    @property
    def vRomFirstOrderMean(self):
        return self._vRomFirstOrderMean

    @vRomFirstOrderMean.setter
    def vRomFirstOrderMean(self, value):
        self._validate_shape(value, (self.vNmodes,), "vRomFirstOrderMean")
        self._vRomFirstOrderMean = value

    @property
    def vRomSecondOrderMat(self):
        return self._vRomSecondOrderMat

    @vRomSecondOrderMat.setter
    def vRomSecondOrderMat(self, value):
        self._validate_shape(value, (self.vNmodes, self.vNmodes), "vRomSecondOrderMat")
        self._vRomSecondOrderMat = value

    @property
    def vRomSecondOrderMean(self):
        return self._vRomSecondOrderMean

    @vRomSecondOrderMean.setter
    def vRomSecondOrderMean(self, value):
        self._validate_shape(value, (self.vNmodes,), "vRomSecondOrderMean")
        self._vRomSecondOrderMean = value

    @property
    def uNonlinDim(self):
        return self._uNonlinDim

    @uNonlinDim.setter
    def uNonlinDim(self, value):
        if not isinstance(value, int):
            raise ValueError("Incorrect type for uNonlinDim: ", type(value))
        elif value < 0 or value > self.uNmodes:
            raise ValueError("Icompatible value for uNonlinDim: ", value)
        else:
            self._uNonlinDim = value

    @property
    def vNonlinDim(self):
        return self._vNonlinDim

    @vNonlinDim.setter
    def vNonlinDim(self, value):
        if not isinstance(value, int):
            raise ValueError("Incorrect type for vNonlinDim: ", type(value))
        elif value < 0 or value > self.vNmodes:
            raise ValueError("Icompatible value for vNonlinDim: ", value)
        else:
            self._vNonlinDim = value

    @property
    def deimProjection(self):
        return self._deimProjection

    @deimProjection.setter
    def deimProjection(self, value):
        self._deimProjection = value

    @property
    def uNonLinProjection(self):
        return self._uNonLinProjection

    @uNonLinProjection.setter
    def uNonLinProjection(self, value):
        self._uNonLinProjection = value

    @property
    def vNonLinProjection(self):
        return self._vNonLinProjection

    @vNonLinProjection.setter
    def vNonLinProjection(self, value):
        self._vNonLinProjection = value

    @property
    def penaltyStrength(self):
        return self._penaltyStrength

    @penaltyStrength.setter
    def penaltyStrength(self, value):
        self._penaltyStrength = value
