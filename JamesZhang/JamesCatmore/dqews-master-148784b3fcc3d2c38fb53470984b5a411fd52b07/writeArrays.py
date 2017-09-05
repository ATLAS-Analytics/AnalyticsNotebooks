import sys
import h5py
import numpy as np
from root_numpy import tree2array, root2array
import ROOT
from sklearn.preprocessing import Imputer
from featuresLists import trackingFeatures

# User-defined
fileName = "304008_lb0775-lb0784.root"
treeName = "DQEWSTree"
selection = ""

# ROOT files
outputName = fileName[:len(fileName)-5]+".h5"
inputFile = ROOT.TFile(fileName)
inputTree = inputFile.Get(treeName)
totalEvents = inputTree.GetEntries()

# Convert to Numpy
treeArray = tree2array(inputTree,
                       start=0,
                       stop=totalEvents,
                       step=1,
                       selection = selection,
                       branches=trackingFeatures)

# Break up into columns
outputArray = np.array(treeArray[:].tolist())

# Replane -999
imp = Imputer(missing_values=-999, strategy='mean', axis=0)
imp.fit(outputArray)
outputArray = imp.transform(outputArray)

print "Number of events    = ",outputArray.shape[0]
print "Number of features  = ",outputArray.shape[1]

f = h5py.File(outputName, "w")
f.create_dataset(treeName, data=outputArray)
f.close()

print "Output file name = ",outputName




