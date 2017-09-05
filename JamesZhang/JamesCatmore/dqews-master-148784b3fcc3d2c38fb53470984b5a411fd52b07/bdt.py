import sys
import pickle

import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from ROOT import gROOT, gDirectory, TFile, TEventList, TCut

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

from CommonTools import reconstructionError,reconstructionErrorByFeature,buildArraysFromROOT,makeMetrics,areaUnderROC
from featuresLists import trackingFeatures 

import logging

logging.basicConfig(
                    format="%(message)s",
                    level=logging.DEBUG,
                    stream=sys.stdout)

pdf_pages = PdfPages('bdt_zerofield.pdf')
np.set_printoptions(threshold=np.nan)

###############################################
# MAIN PROGRAM

runTraining = True

# Selections
#cutBackground = "isSignal==0"
#cutSignal = "isSignal==1 && massSplit>499 && massSplit<699"

# Input files and TTrees
referenceFile = TFile("304008_lb0775-lb0784.root","read")
referenceTree = referenceFile.Get("DQEWSTree")
newFile = TFile("299278.root","read")
newTree = newFile.Get("DQEWSTree")

# Automatic numbers of event 
#tcut = TCut(cutSignal)
#tree.Draw(">>eventList",tcut)
#eventList = TEventList()
#eventList = gDirectory.Get("eventList")
#nReferenceEvents = referenceTree.GetEntries()/2
#nNewEvents = newTree.GetEntries()/2
nReferenceEvents = 100000
nNewEvents = 100000


# Build data arrays
# Reference
X_train_ref = buildArraysFromROOT(referenceTree,trackingFeatures,"",0,nReferenceEvents,"TRAINING SAMPLE (reference)")
X_test_ref = buildArraysFromROOT(referenceTree,trackingFeatures,"",nReferenceEvents,nReferenceEvents,"TESTING SAMPLE (reference)")

# New data
X_train_new = buildArraysFromROOT(newTree,trackingFeatures,"",0,nNewEvents,"TRAINING SAMPLE (new)")
X_test_new = buildArraysFromROOT(newTree,trackingFeatures,"",nNewEvents,nNewEvents,"TESTING SAMPLE (new)")

# Combining signal & background
X_train = np.concatenate((X_train_ref,X_train_new),0)
X_test = np.concatenate((X_test_ref,X_test_new),0)
Y_ref = np.zeros(nReferenceEvents)
Y_new = np.ones(nNewEvents)
Y = np.concatenate((Y_ref,Y_new),0)

# Feature scaling
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)

# BDT TRAINING AND TESTING
print "Building and training BDT"
clf = AdaBoostClassifier(n_estimators=100,base_estimator=DecisionTreeClassifier(max_depth=1))
clf.fit(X_train,Y)

# Testing
pred_train = clf.predict(X_train)
pred_test = clf.predict(X_test)
output_train = clf.decision_function(X_train)
output_test = clf.decision_function(X_test)

print "Training sample...."
print "  Signal identified as signal (%)        : ",100.0*np.sum(pred_train[nReferenceEvents:nReferenceEvents+nNewEvents]==1.0)/nNewEvents
print "  Signal identified as background (%)    : ",100.0*np.sum(pred_train[nReferenceEvents:nReferenceEvents+nNewEvents]==0.0)/nNewEvents
print "  Background identified as signal (%)    : ",100.0*np.sum(pred_train[0:nReferenceEvents]==1.0)/nReferenceEvents
print "  Background identified as background (%): ",100.0*np.sum(pred_train[0:nReferenceEvents]==0.0)/nReferenceEvents
print ""
print "Testing sample...."
print "  Signal identified as signal (%)        : ",100.0*np.sum(pred_test[nReferenceEvents:nReferenceEvents+nNewEvents]==1.0)/nNewEvents
print "  Signal identified as background (%)    : ",100.0*np.sum(pred_test[nReferenceEvents:nReferenceEvents+nNewEvents]==0.0)/nNewEvents
print "  Background identified as signal (%)    : ",100.0*np.sum(pred_test[0:nReferenceEvents]==1.0)/nReferenceEvents
print "  Background identified as background (%): ",100.0*np.sum(pred_test[0:nReferenceEvents]==0.0)/nReferenceEvents

# Plotting - probabilities
#print output_train[(Y==0.0).reshape(2*nEvents,)]

figA, axsA = plt.subplots(2, 1)
ax1, ax2 = axsA.ravel()
for ax in ax1, ax2:
    ax.set_ylabel("Events")
    ax.set_xlabel("NN signal probability")
bins = np.linspace(-0.5, 1.5, 250)
ax1.hist(output_train[(Y==0.0).reshape(nReferenceEvents+nNewEvents,)], bins, facecolor='blue', alpha=0.4, histtype='stepfilled')
ax1.hist(output_train[(Y==1.0).reshape(nReferenceEvents+nNewEvents,)], bins, facecolor='red', alpha=0.4, histtype='stepfilled')
ax2.hist(output_test[(Y==0.0).reshape(nReferenceEvents+nNewEvents,)], bins, facecolor='green', alpha=0.4, histtype='stepfilled')
ax2.hist(output_test[(Y==1.0).reshape(nReferenceEvents+nNewEvents,)], bins, facecolor='red', alpha=0.4, histtype='stepfilled')
pdf_pages.savefig(figA)


# Plotting - performance curves
# ROC
fpr, tpr, thresholds = roc_curve(Y, output_test, pos_label=1)
auc = roc_auc_score(Y, output_test)
print "Area under ROC = ",auc
figB, axB1 = plt.subplots()
#axB1,axB2 = axsB.ravel()
axB1.plot(fpr, tpr, label='ROC curve')
axB1.plot([0, 1], [0, 1], 'k--')
axB1.set_xlim([0.0, 1.0])
axB1.set_ylim([0.0, 1.05])
axB1.set_xlabel('False Signal Rate')
axB1.set_ylabel('True Signal Rate')
axB1.text(0.4,0.2,"AUC = %.4f" % auc,fontsize=15)
pdf_pages.savefig(figB)

# Variable importances
y_pos = np.arange(len(trackingFeatures))
figC, axC1 = plt.subplots(1,1)
axC1.barh(y_pos, 100.0*clf.feature_importances_, align='center', alpha=0.4)
axC1.set_ylim([0,len(trackingFeatures)])
axC1.set_yticks(y_pos)
axC1.set_yticklabels(trackingFeatures,fontsize=3)
axC1.set_xlabel('Relative importance, %')
axC1.set_title("Estimated variable importance using outputs (BDT)")
pdf_pages.savefig(figC)


## Plot everything
#plt.show()
pdf_pages.close()
