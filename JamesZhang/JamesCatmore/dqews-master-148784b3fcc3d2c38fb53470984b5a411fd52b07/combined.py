import sys
import pickle

# NumPy
import numpy as np

# HDF5
import h5py

# Matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# SKL
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# Keras and Tensorflow
from keras.models import Sequential, load_model

# Home-made tools, mainly for reading ROOT files
from CommonTools import reconstructionError,relativeErrorByFeature,printResults

# Feature list
from featuresLists import trackingFeatures

# Logging
import logging
logging.basicConfig(
                    format="%(message)s",
                    level=logging.DEBUG,
                    stream=sys.stdout)

# Output PDF (book of plots)
pdf_pages = PdfPages('queenbork.pdf')
np.set_printoptions(threshold=np.nan)

###############################################
# MAIN PROGRAM
#############################

runTraining = True
runAE = True
runNN = True
runBDT = True
nReferenceEvents = 100000
nNewEvents = 100000

#############################
# Reading in data and pre-processing
#############################

# Input files
referenceFile = h5py.File("304008_lb0775-lb0784.h5","r")
newFile = h5py.File("304008_lb0785-lb0794.h5","r")

# Get numpy arrays
# Reference
X_train_ref = referenceFile['DQEWSTree'][0:nReferenceEvents]
X_test_ref = referenceFile['DQEWSTree'][nReferenceEvents:nReferenceEvents+nReferenceEvents]
referenceFile.close()

# New data
X_train_new = newFile['DQEWSTree'][0:nNewEvents]
X_test_new = newFile['DQEWSTree'][nNewEvents:nNewEvents+nNewEvents]
newFile.close()

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
# Special feature scaling for autoencoder (since it is only trained on the reference)
min_max_scaler_ae = preprocessing.MinMaxScaler()
X_train_ref_ae = min_max_scaler_ae.fit_transform(X_train_ref)
X_test_ref_ae = min_max_scaler.transform(X_test_ref)
X_test_new_ae = min_max_scaler.transform(X_test_new)

#############################
# Assembly, training & testing
#############################

# Boosted decision tree classifier
if runBDT:
    print "Building and training BDT"
    bdt = AdaBoostClassifier(n_estimators=100,base_estimator=DecisionTreeClassifier(max_depth=1))
    bdt.fit(X_train,Y)
    # Testing
    pred_train_bdt = bdt.predict(X_train)
    pred_test_bdt = bdt.predict(X_test)
    output_train_bdt = bdt.decision_function(X_train)
    output_test_bdt = bdt.decision_function(X_test)
    # Results print-out
    print "BDT classifier esults...."
    printResults(pred_train_bdt,pred_test_bdt,nReferenceEvents,nNewEvents)

# Neural network classifier
if runNN:
    # Training
    if runTraining:
        print "Building and training neural network"
        nn = Sequential()
        from keras.layers import Dense, Activation
        nn.add(Dense(71, input_dim=71))
        nn.add(Activation("relu"))
        nn.add(Dense(1))
        nn.add(Activation("sigmoid"))
        nn.compile(optimizer="sgd",
                   loss="mean_squared_error",
                   class_mode="binary",
                   metrics=["mean_squared_error"])
        nn.fit(X_train,Y,epochs=100,batch_size=100)
        nn.save("nn_classifier.h5")
    if not runTraining:
        print "Reading in pre-trained neural network"
        nn = load_model("nn_classifier.h5")
    # Testing
    pred_train_nn = nn.predict_classes(X_train,batch_size=100)
    pred_test_nn = nn.predict_classes(X_test,batch_size=100)
    probabilities_train_nn = nn.predict_proba(X_train,batch_size=100)
    probabilities_test_nn = nn.predict_proba(X_test,batch_size=100)
    print "NN classifier results...."
    printResults(pred_train_nn,pred_test_nn,nReferenceEvents,nNewEvents)

# Autoencoder
if runAE:
    if runTraining:
        print "Building and training autoencoder"
        ae = Sequential()
        from keras.layers import Dense, Activation
        ae.add(Dense(units=20, input_dim=71))
        ae.add(Activation("relu"))
        ae.add(Dense(units=71, input_dim=20))
        ae.add(Activation("linear"))
        ae.compile(loss="mean_squared_error",optimizer="sgd")
        ae.fit(X_train_ref_ae,X_train_ref_ae,epochs=100,batch_size=100)
        ae.save("autoencoder.h5")
    if not runTraining:
        print "Reading pre-trained autoencoder"
        ae = load_model("autoencoder.h5")
    # Testing
    predicted_same_ae = ae.predict_proba(X_train_ref_ae)
    predicted_diff_ae = ae.predict_proba(X_test_ref_ae)
    predicted_new_ae = ae.predict_proba(X_test_new_ae)
    # Reconstruction error
    rec_errors_same_ae = reconstructionError(X_train_ref_ae,predicted_same_ae)
    rec_errors_diff_ae = reconstructionError(X_test_ref_ae,predicted_diff_ae)
    rec_errors_new_ae = reconstructionError(X_test_new_ae,predicted_new_ae)


#############################
# Plotting
#############################
y_pos = np.arange(len(trackingFeatures))
bins = np.linspace(-0.5, 1.5, 250)

if runBDT:
    # Outputs
    figBDT, axsBDT = plt.subplots(2, 1)
    axBDT1, axBDT2 = axsBDT.ravel()
    for ax in axBDT1, axBDT2:
        ax.set_ylabel("Events")
        ax.set_xlabel("BDT output")
    axBDT1.hist(output_train_bdt[(Y==0.0).reshape(nReferenceEvents+nNewEvents,)], bins, facecolor='blue', alpha=0.4, histtype='stepfilled')
    axBDT1.hist(output_train_bdt[(Y==1.0).reshape(nReferenceEvents+nNewEvents,)], bins, facecolor='red', alpha=0.4, histtype='stepfilled')
    axBDT2.hist(output_test_bdt[(Y==0.0).reshape(nReferenceEvents+nNewEvents,)], bins, facecolor='green', alpha=0.4, histtype='stepfilled')
    axBDT2.hist(output_test_bdt[(Y==1.0).reshape(nReferenceEvents+nNewEvents,)], bins, facecolor='red', alpha=0.4, histtype='stepfilled')

    pdf_pages.savefig(figBDT)
    # ROC
    fpr, tpr, thresholds = roc_curve(Y, output_test_bdt, pos_label=1)
    bdt_auc = roc_auc_score(Y, output_test_bdt)
    figBDTROC, axBDTROC = plt.subplots()
    #axB1,axB2 = axsB.ravel()
    axBDTROC.plot(fpr, tpr, label='ROC curve')
    axBDTROC.plot([0, 1], [0, 1], 'k--')
    axBDTROC.set_xlim([0.0, 1.0])
    axBDTROC.set_ylim([0.0, 1.05])
    axBDTROC.set_xlabel('False Signal Rate')
    axBDTROC.set_ylabel('True Signal Rate')
    axBDTROC.text(0.4,0.2,"AUC = %.4f" % bdt_auc,fontsize=15)
    axBDTROC.set_title("BDT ROC")
    pdf_pages.savefig(figBDTROC)
    # Variable importances
    figVarsBDT, axVarsBDT = plt.subplots(1,1)
    axVarsBDT.barh(y_pos, 100.0*bdt.feature_importances_, align='center', alpha=0.4)
    axVarsBDT.set_ylim([0,len(trackingFeatures)])
    axVarsBDT.set_yticks(y_pos)
    axVarsBDT.set_yticklabels(trackingFeatures,fontsize=3)
    axVarsBDT.set_xlabel('Relative importance, %')
    axVarsBDT.set_title("Estimated BDT variable importance")
    pdf_pages.savefig(figVarsBDT)

if runNN:
    # Outputs
    figNN, axsNN = plt.subplots(2, 1)
    axNN1, axNN2 = axsNN.ravel()
    for ax in axNN1, axNN2:
        ax.set_ylabel("Events")
        ax.set_xlabel("NN output")
    axNN1.hist(probabilities_train_nn[(Y==0.0).reshape(nReferenceEvents+nNewEvents,)], bins, facecolor='blue', alpha=0.4, histtype='stepfilled')
    axNN1.hist(probabilities_train_nn[(Y==1.0).reshape(nReferenceEvents+nNewEvents,)], bins, facecolor='red', alpha=0.4, histtype='stepfilled')
    axNN2.hist(probabilities_test_nn[(Y==0.0).reshape(nReferenceEvents+nNewEvents,)], bins, facecolor='green', alpha=0.4, histtype='stepfilled')
    axNN2.hist(probabilities_test_nn[(Y==1.0).reshape(nReferenceEvents+nNewEvents,)], bins, facecolor='red', alpha=0.4, histtype='stepfilled')
    pdf_pages.savefig(figNN)
    # ROC
    fpr, tpr, thresholds = roc_curve(Y, probabilities_test_nn, pos_label=1)
    nn_auc = roc_auc_score(Y, probabilities_test_nn)
    figRocNN, axRocNN = plt.subplots()
    #axB1,axB2 = axsB.ravel()
    axRocNN.plot(fpr, tpr, label='ROC curve')
    axRocNN.plot([0, 1], [0, 1], 'k--')
    axRocNN.set_xlim([0.0, 1.0])
    axRocNN.set_ylim([0.0, 1.05])
    axRocNN.set_xlabel('False Signal Rate')
    axRocNN.set_ylabel('True Signal Rate')
    axRocNN.text(0.4,0.2,"AUC = %.4f" % nn_auc,fontsize=15)
    axRocNN.set_title("NN ROC")
    pdf_pages.savefig(figRocNN)
    # Assess variable importance using weights method
    weights = np.array([])
    for layer in nn.layers:
        if layer.name =="dense_1":
            weights = layer.get_weights()[0]
    # Ecol. Modelling 160 (2003) 249-264
    sumWeights = np.sum(np.absolute(weights),axis=0)
    Q = np.absolute(weights)/sumWeights
    R = 100.0 * np.sum(Q,axis=1) / np.sum(np.sum(Q,axis=0))
    figVarsNN, axVarsNN = plt.subplots()
    axVarsNN.barh(y_pos, R, align='center', alpha=0.4)
    axVarsNN.set_ylim([0,len(trackingFeatures)])
    axVarsNN.set_yticks(y_pos)
    axVarsNN.set_yticklabels(trackingFeatures,fontsize=3)
    axVarsNN.set_xlabel('Relative importance, %')
    axVarsNN.set_title("Estimated NN variable importance using input-hidden weights (ecol.model)")
    pdf_pages.savefig(figVarsNN)

if runAE:
    # Outputs
    figAE, axsAE = plt.subplots(2, 1)
    axAE1, axAE2 = axsAE.ravel()
    for ax in axAE1, axAE2:
        ax.set_ylabel("Events")
        ax.set_xlabel("log10(Autoencoder reconstruction error)")
    axAE1.hist(rec_errors_same_ae, bins, facecolor='blue', alpha=0.4, histtype='stepfilled')
    axAE1.hist(rec_errors_diff_ae, bins, facecolor='green', alpha=0.4, histtype='stepfilled')
    axAE2.hist(rec_errors_diff_ae, bins, facecolor='green', alpha=0.4, histtype='stepfilled')
    axAE2.hist(rec_errors_new_ae, bins, facecolor='red', alpha=0.4, histtype='stepfilled')
    pdf_pages.savefig(figAE)
    # ROC
    Y_ref = np.zeros(nReferenceEvents).reshape(nReferenceEvents,1)
    Y_new = np.ones(nNewEvents).reshape(nNewEvents,1)
    Y_ROC = np.concatenate((Y_ref,Y_new),0)
    rec_errors_ROC = np.concatenate((rec_errors_diff_ae,rec_errors_new_ae),0)
    fpr, tpr, thresholds = roc_curve(Y_ROC, rec_errors_ROC, pos_label=1)
    ae_auc = roc_auc_score(Y_ROC, rec_errors_ROC)
    figRocAE, axRocAE = plt.subplots()
    axRocAE.plot(fpr, tpr, label='ROC curve')
    axRocAE.plot([0, 1], [0, 1], 'k--')
    axRocAE.set_xlim([0.0, 1.0])
    axRocAE.set_ylim([0.0, 1.05])
    axRocAE.set_xlabel('False Anomaly Rate')
    axRocAE.set_ylabel('True Anomaly Rate')
    axRocAE.text(0.4,0.2,"AUC = %.4f" % ae_auc,fontsize=15)
    axRocAE.set_title("Autoencoder ROC")
    pdf_pages.savefig(figRocAE)
    # Variable importance
    rec_errors_varwise_new = relativeErrorByFeature(X_test_new_ae,predicted_new_ae)
    figVarAE, axVarAE = plt.subplots(1,1)
    axVarAE.barh(y_pos, 100*rec_errors_varwise_new, align='center', alpha=0.4)
    axVarAE.set_ylim([0,len(trackingFeatures)])
    axVarAE.set_yticks(y_pos)
    axVarAE.set_yticklabels(trackingFeatures,fontsize=3)
    axVarAE.set_xlabel('Relative importance, %')
    axVarAE.set_title("Estimated variable importance using outputs (autoencoder)")
    pdf_pages.savefig(figVarAE)

print ""
print "#########################"
print "Areas under ROCs..."
if runBDT: print "BDT         : ",bdt_auc
if runNN: print "NN          : ",nn_auc
if runAE: print "Autoencoder : ",ae_auc
print "#########################"


## Plot everything
#plt.show()
pdf_pages.close()
