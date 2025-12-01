import os

import matplotlib.pyplot as plt
import numpy as np

from tankModel.TankModel import TankModel

nCollocations = [1, 2, 3]
verbosity = 1

# I think there's an error with the higher
nElems = [
    2,
    4,
    8,
    16,
    32,
    64,
    128,
]  # Cant use nElems=1 due to some dimensionality issues with squeeze
parameterSet = "Bizon2012_stable"
# Parameter limitations:
# Non-negative: Da, gamma, beta, delta
# Positive: Le, PeM,
# Other: f can be any number except 1
# Parameter Relations
# If delta=0: then vH has no effect
# If Da=0: then beta and gamma have no effect
# If Da=0 and delta=0: then Le has no effect

# Parameter set with improved conditioning
if parameterSet == "Bizon2012_improvedConditioning":
    params = {
        "PeM": 1000,
        "PeT": 1000,
        "f": 0.3,
        "Le": 1,
        "Da": 0.15,
        "beta": 1.4,
        "gamma": 10,
        "delta": 2,
        "vH": -0.2,
    }
# Bizon2012 Parameters for  a stable domain
elif parameterSet == "Bizon2012_stable":
    params = {
        "PeM": 300,
        "PeT": 300,
        "f": 0.3,
        "Le": 1,
        "Da": 0.15,
        "beta": 1.4,
        "gamma": 10,
        "delta": 2,
        "vH": -0.2,
    }
# Bizon2012 Parameters without nonlinear effects
elif parameterSet == "Bizon2012_linear":
    params = {
        "PeM": 300,
        "PeT": 300,
        "f": 0.3,
        "Le": 1,
        "Da": 0,
        "beta": 0,
        "gamma": 0,
        "delta": 2,
        "vH": -0.2,
    }
# Bizon2012 Parameters with just diffusion/advection
elif parameterSet == "Bizon2012_diffAdvec":
    params = {
        "PeM": 300,
        "PeT": 300,
        "f": 0,
        "Le": 1,
        "Da": 0,
        "beta": 0,
        "gamma": 0,
        "delta": 0,
        "vH": 0,
    }
# Unit parameters with just diffusion/advection
# params={"PeM": 1, "PeT": 1, "f": .5, "Le": 1, "Da": 0, "beta": 0, "gamma": 0,"delta": 0, "vH": 0}

resultsFolder = "../../results/verification/"

massBoundaryConditionNumber = np.empty((len(nCollocations), len(nElems)))
tempBoundaryConditionNumber = np.empty((len(nCollocations), len(nElems)))

saveLocation = resultsFolder + "/" + parameterSet + "/CondNumber/"
if not os.path.exists(saveLocation):
    os.makedirs(saveLocation)
for iColl in range(len(nCollocations)):
    for iElem in range(len(nElems)):
        model = TankModel(
            nCollocation=nCollocations[iColl],
            nElements=nElems[iElem],
            spacing="legendre",
            bounds=[0, 1],
            params=params,
            verbosity=verbosity,
        )
        massBoundaryConditionNumber[iColl, iElem] = np.linalg.cond(model.massBoundaryMat)
        tempBoundaryConditionNumber[iColl, iElem] = np.linalg.cond(model.tempBoundaryMat)

print(["$N_C=$" + str(nColl) for nColl in nCollocations])
plt.figure(figsize=(5.12, 3.84))
plt.semilogx(nElems, massBoundaryConditionNumber.transpose())
plt.semilogx(nElems, tempBoundaryConditionNumber.transpose(), "ks")
plt.xlabel(r"$N_E$")
plt.ylabel(r"$\mathbf{B}^{(u)}$ Condition Number")
plt.legend(["$N_C=$" + str(nColl) for nColl in nCollocations])
plt.tight_layout()
plt.savefig(saveLocation + "u.png")
plt.savefig(saveLocation + "u.pdf")

plt.figure(figsize=(5.12, 3.84))
plt.semilogx(nElems, tempBoundaryConditionNumber.transpose())
plt.semilogx(nElems, tempBoundaryConditionNumber.transpose(), "ks")
plt.xlabel(r"$N_E$")
plt.ylabel(r"$\mathbf{B}^{(v)}$ Condition Number")
plt.legend(["$N_C=$" + str(nColl) for nColl in nCollocations])
plt.tight_layout()
plt.savefig(saveLocation + "v.png")
plt.savefig(saveLocation + "v.pdf")
plt.show()
