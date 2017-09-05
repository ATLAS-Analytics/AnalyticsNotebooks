import sys
import pickle

from ROOT import gROOT, gDirectory, TFile, TEventList, TCut

import numpy as np
from keras.models import Sequential, load_model
from sklearn import preprocessing

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve


from CommonTools import reconstructionError,relativeErrorByFeature,buildArraysFromROOT
from featuresLists import trackingFeatures

import logging

logging.basicConfig(
                    format="%(message)s",
                    level=logging.DEBUG,
                    stream=sys.stdout)

pdf_pages = PdfPages('autoencoder_pixelscan.pdf')

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
#nNewEvents = eventList.GetN()/2
nNewEvents = 100000
nReferenceEvents = nNewEvents

# Assemble data arrays
X_train = buildArraysFromROOT(referenceTree,trackingFeatures,"",0,nReferenceEvents,"TRAINING SAMPLE (reference only)")
X_test = buildArraysFromROOT(referenceTree,trackingFeatures,"",nReferenceEvents,nReferenceEvents,"TESTING SAMPLE - reference")
X_new = buildArraysFromROOT(newTree,trackingFeatures,"",0,nNewEvents,"TESTING SAMPLE - new")

# Feature scaling
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)
X_new = min_max_scaler.transform(X_new)

# Set target equal to input - autoencoder 
Y_train = X_train
print X_train.shape

# NEURAL NETWORK TRAINING AND TESTING
# Set up neural network
if runTraining:
    print "Building and training autoencoder"
    model = Sequential()
    from keras.layers import Dense, Activation
    model.add(Dense(output_dim=20, input_dim=71))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=71, input_dim=20))
    model.add(Activation("linear"))
    model.compile(loss="mean_squared_error",optimizer="sgd")
    model.fit(X_train,Y_train,nb_epoch=100,batch_size=100)
    model.save("autoencoder.h5")

if not runTraining:
    print "Reading pre-trained autoencoder"
    model = load_model("autoencoder.h5")

# Testing
predicted_same = model.predict_proba(X_train)
predicted_diff = model.predict_proba(X_test)
predicted_new = model.predict_proba(X_new)

# Reconstruction error
rec_errors_same = reconstructionError(X_train,predicted_same)
rec_errors_diff = reconstructionError(X_test,predicted_diff)
rec_errors_new = reconstructionError(X_new,predicted_new)

# Reconstruction errors by variable
rec_errors_varwise_same = relativeErrorByFeature(X_train,predicted_same)
rec_errors_varwise_diff = relativeErrorByFeature(X_test,predicted_diff)
rec_errors_varwise_new = relativeErrorByFeature(X_new,predicted_new)

# Plotting - reconstruction errors
fig, axs = plt.subplots(3, 1)
ax1, ax2, ax3 = axs.ravel()
for ax in ax1, ax2, ax3:
    ax.set_ylabel("Events")
    ax.set_xlabel("log10(Reconstruction error)")
bins = np.linspace(-2.0, 5.0, 250)
ax1.hist(rec_errors_same, bins, facecolor='blue', alpha=0.4, histtype='stepfilled')
ax2.hist(rec_errors_diff, bins, facecolor='green', alpha=0.4, histtype='stepfilled')
ax3.hist(rec_errors_diff, bins, facecolor='green', alpha=0.4, histtype='stepfilled')
ax3.hist(rec_errors_new, bins, facecolor='red', alpha=0.4, histtype='stepfilled')
pdf_pages.savefig(fig)


## 2D plots
#fig2DSig, ax2DSig = plt.subplots(7,7)
#i=0
#for ax in ax2DSig.ravel():
#    ax.hist2d(X_new[:,i:i+1].reshape([20000,]),rec_errors_new, bins=100)
#    i=i+1
#fig2DBg, ax2DBg = plt.subplots(7,7)
#i=0
#for ax in ax2DBg.ravel():
#    ax.hist2d(X_test[:,i:i+1].reshape([20000,]),rec_errors_diff, bins=100)
#    i=i+1
#

# Plotting - performance curve (ROC)
Y_ref = np.zeros(nReferenceEvents).reshape(nReferenceEvents,1)
Y_new = np.ones(nNewEvents).reshape(nNewEvents,1)
Y_ROC = np.concatenate((Y_ref,Y_new),0)
rec_errors_ROC = np.concatenate((rec_errors_diff,rec_errors_new),0)
fpr, tpr, thresholds = roc_curve(Y_ROC, rec_errors_ROC, pos_label=1)
auc = roc_auc_score(Y_ROC, rec_errors_ROC)
print ""
print "Area under ROC = ",auc
figB, axB1 = plt.subplots()
axB1.plot(fpr, tpr, label='ROC curve')
axB1.plot([0, 1], [0, 1], 'k--')
axB1.set_xlim([0.0, 1.0])
axB1.set_ylim([0.0, 1.05])
axB1.set_xlabel('False Anomaly Rate')
axB1.set_ylabel('True Anomaly Rate')
axB1.text(0.4,0.2,"AUC = %.4f" % auc,fontsize=15)
pdf_pages.savefig(figB)


# Variable importance
y_pos = np.arange(len(trackingFeatures))
figC, axC1 = plt.subplots(1,1)
axC1.barh(y_pos, 100*rec_errors_varwise_new, align='center', alpha=0.4)
axC1.set_ylim([0,len(trackingFeatures)])
axC1.set_yticks(y_pos)
axC1.set_yticklabels(trackingFeatures,fontsize=3)
axC1.set_xlabel('Relative importance, %')
axC1.set_title("Estimated variable importance using outputs (autoencoder)")
pdf_pages.savefig(figC)

pdf_pages.close()
#plt.show()









