import sys
import pickle

import numpy as np
from keras.models import Sequential, load_model
from sklearn import preprocessing

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

pdf_pages = PdfPages('classifier_pixelscan.pdf')
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
newFile = TFile("296942.root","read")
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
Y_ref = np.zeros(nReferenceEvents).reshape(nReferenceEvents,1)
Y_new = np.ones(nNewEvents).reshape(nNewEvents,1)
Y = np.concatenate((Y_ref,Y_new),0)

# Feature scaling
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)

# NEURAL NETWORK TRAINING AND TESTING
# Set up neural network
if runTraining:
    print "Building and training neural network"
    model = Sequential()
    from keras.layers import Dense, Activation
    model.add(Dense(71, input_dim=71))
    model.add(Activation("relu"))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    model.compile(optimizer="sgd",
                  loss="mean_squared_error",
                  class_mode="binary",
                  metrics=["mean_squared_error"])
    model.fit(X_train,Y,epochs=100,batch_size=100)
    model.save("classifier.h5")

if not runTraining:
    print "Reading in pre-trained neural network"
    model = load_model("classifier.h5")

# Testing
pred_train = model.predict_classes(X_train,batch_size=100)
pred_test = model.predict_classes(X_test,batch_size=100)
probabilities_train = model.predict_proba(X_train,batch_size=100)
probabilities_test = model.predict_proba(X_test,batch_size=100)


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
#print probabilities_train[(Y==0.0).reshape(2*nEvents,)]

figA, axsA = plt.subplots(2, 1)
ax1, ax2 = axsA.ravel()
for ax in ax1, ax2:
    ax.set_ylabel("Events")
    ax.set_xlabel("NN signal probability")
bins = np.linspace(-0.5, 1.5, 250)
ax1.hist(probabilities_train[(Y==0.0).reshape(nReferenceEvents+nNewEvents,)], bins, facecolor='blue', alpha=0.4, histtype='stepfilled')
ax1.hist(probabilities_train[(Y==1.0).reshape(nReferenceEvents+nNewEvents,)], bins, facecolor='red', alpha=0.4, histtype='stepfilled')
ax2.hist(probabilities_test[(Y==0.0).reshape(nReferenceEvents+nNewEvents,)], bins, facecolor='green', alpha=0.4, histtype='stepfilled')
ax2.hist(probabilities_test[(Y==1.0).reshape(nReferenceEvents+nNewEvents,)], bins, facecolor='red', alpha=0.4, histtype='stepfilled')
pdf_pages.savefig(figA)


# Plotting - performance curves
# ROC
fpr, tpr, thresholds = roc_curve(Y, probabilities_test, pos_label=1)
auc = roc_auc_score(Y, probabilities_test)
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

# Assess variable importance using weights method
weights = np.array([])
for layer in model.layers:
    if layer.name =="dense_1":
        weights = layer.get_weights()[0]
# Ecol. Modelling 160 (2003) 249-264
sumWeights = np.sum(np.absolute(weights),axis=0)
Q = np.absolute(weights)/sumWeights
R = 100.0 * np.sum(Q,axis=1) / np.sum(np.sum(Q,axis=0))
y_pos = np.arange(len(trackingFeatures))
figC, axC = plt.subplots()
axC.barh(y_pos, R, align='center', alpha=0.4)
axC.set_ylim([0,len(trackingFeatures)])
axC.set_yticks(y_pos)
axC.set_yticklabels(trackingFeatures,fontsize=3)
axC.set_xlabel('Relative importance, %')
axC.set_title('Estimated variable importance using input-hidden weights (ecol.model)')
pdf_pages.savefig(figC)


# Plot variables
#for j in range(0,len(trackingFeatures)):
#    fig, ax = plt.subplots()
#    min = np.array([X_train[:,j].min(),X_test[:,j].min()]).min()
#    max = np.array([X_train[:,j].max(),X_test[:,j].max()]).max()
#    theseBins = np.linspace(min, max, 250)
#    ax.hist(X_train[:,j], theseBins, facecolor='blue', alpha=0.4, histtype='stepfilled')
#    ax.hist(X_test[:,j], theseBins, facecolor='red', alpha=0.4, histtype='stepfilled')
#    ax.set_xlabel(trackingFeatures[j])
#    pdf_pages.savefig(fig)


## Plot everything
#plt.show()
pdf_pages.close()
