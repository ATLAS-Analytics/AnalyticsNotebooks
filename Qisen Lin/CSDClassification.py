#! /usr/bin/env python
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn import metrics
# fix random seed for reproducibility
np.random.seed(7)



dataset = np.loadtxt("data_train.txt", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:78]
Y = dataset[:,78]

model=Sequential();
model.add(Dense(256, input_dim=78, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
#model.add(Dense(128, activation='relu'))
#model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#model.fit(X, Y, epochs=150, batch_size=10,verbose=0)
model.fit(X, Y, epochs=800, batch_size=128)

scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

dataset_test = np.loadtxt("data_test.txt", delimiter=",")
# split into input (X) and output (Y) variables
X_test = dataset_test[:,0:78]
Y_test = dataset_test[:,78]

probs = model.predict_proba(X_test)
np.savetxt('testout.txt',probs,delimiter=',')
preds = probs[:,0]
fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

dataset_application_s=np.loadtxt('data_s.txt',delimiter=",")
X_s=dataset_application_s[:,0:78]
score_s=model.predict_proba(X_s)
np.savetxt('s_socre.txt', score_s,delimiter=',')

dataset_application_b=np.loadtxt('data_b.txt',delimiter=",")
X_b=dataset_application_b[:,0:78]
score_b=model.predict_proba(X_b)
np.savetxt('b_socre.txt', score_b,delimiter=',')


# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(tpr, 1-fpr, 'b') 
#plt.plot(tpr, fpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
#plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('signal efficiency')
plt.ylabel('background rejection')
plt.show()
