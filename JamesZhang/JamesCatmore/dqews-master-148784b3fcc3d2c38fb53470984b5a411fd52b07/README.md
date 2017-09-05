# Data Quality Early Warning System - R&D package

This package is for R&D on the data quality early warning system, which is intended to be a machine-learning driven tool to alert data quality shifters and experts to data features that appear unusual. The package is just for trying out ideas, and in particular is independent of the ATLAS software (everything is written in python, using numpy arrays, and makes use of Keras and SciKitLearn, and MatPlotLib for plotting).

In the description below the text refers to "reference" and "subject" data, where the reference data is already somehow assumed to be good, and the subject is the data to be tested. If the algorithm makes significant progress in separating the reference and subject, they must be distinguishable by construction and therefore the subject cannot be consistent with the background. 

## Project contents

There are several elements to the package:
1. tools for converting ROOT files (n-tuples or xAOD) to NumPy arrays - `CommonTools.py`
2. program using an auto-encoder as the means of detecting unusual features - `autoencoder.py` 
3. program using a conventional classifier (simple neural network in this case) - `classifier.py`
4. program using a BDT - `bdt.py`
5. program using all three of the above algorithms, to save time reading in the large datasets - `combined.py`
6. list of variable names as found in the ROOT input n-tuple - `featuresList.py`. Currently this is a list of tracking variables.

ROOT and NumPy are used to read and store the data, Keras and Tensorflow are used to build and run the neural networks. SciKitLearn is used to produce the figures of merit, e.g. the ROC curves and false-error rates, as well as implementing the BDT.

## Installation of the pre-requisites
The pre-requisites are:
- Python
- [Numpy](http://www.numpy.org) (data arrays)
- [Scipy](https://www.scipy.org)
- [Matplotlib](http://matplotlib.org) (graphics) (or ROOT if you prefer)
- [SciKit-Learn](http://scikit-learn.org/stable/) (machine learning tool-kit)
- [h5py](http://www.h5py.org) (persistency for NumPy arrays and Keras weights)
- [Keras](https://keras.io) (toolkit for neural network building)
- [TensorFlow](https://www.tensorflow.org) (linear algebra and minimization back-end)
- [Theano](http://deeplearning.net/software/theano/) (linear algebra and minimization back-end)
- [ROOT](https://root.cern.ch) (needed for reading in n-tuples or xAOD files)
- [root\_numpy](http://scikit-hep.org/root_numpy/) (needed for fast conversion of ROOT TTrees in to NumPy arrays))

In principle you only need one of TensorFlow or Theano, and Keras uses TensorFlow by default. The [Anaconda](https://www.continuum.io) package ships most of the above toegther, but not TensorFlow/Theano and Keras.

### Dedicated installation instructions for Mac

First you should install pip. This can be obtained by downloading [get-pip.py](https://bootstrap.pypa.io/get-pip.py) and then

    [sudo] python get-pip.py

To avoid issues with Python versions, you should now install and set up ``virtualenv`` ([link](https://virtualenv.pypa.io/en/stable/)) to set up a ring-fenced Python instance.

    [sudo] pip install virtualenv
    cd $HOME
    virtualenv ENV

where ``ENV`` is whatever name you wish to give to the virtualenv location. This is the last time you should need to use ``sudo``: from now on you will not be touching the system-wide configuration.

Once this is done you should install the above packages using the ``pip`` version available in the virtualenv, rather than the system version that you installed before e.g.

    $HOME/ENV/bin/pip install --upgrade numpy
    $HOME/ENV/bin/pip install --upgrade scipy
    $HOME/ENV/bin/pip install --upgrade matplotlib
    $HOME/ENV/bin/pip install --upgrade scikit-learn
    $HOME/ENV/bin/pip install --upgrade keras
    $HOME/ENV/bin/pip install --upgrade tensorflow

When running the scripts, you should also use the python version in the virtualenv rather than the system version, e.g.

    $HOME/ENV/bin/python autoencoder.py

If you are using the non-ROOT graphics package matplotlib, there is one extra issue, namely that the GUI is incompatible with the virtualenv. To get around this add the following script (call it ``frameworkpython``) to ``$HOME/ENV/bin``:

    #!/bin/bash

    # what real Python executable to use
    PYVER=2.7
    PATHTOPYTHON=/usr/local/bin/
    PYTHON=${PATHTOPYTHON}python${PYVER}

    # find the root of the virtualenv, it should be the parent of the dir this script is in
    ENV=`$PYTHON -c "import os; print os.path.abspath(os.path.join(os.path.dirname(\"$0\"), '..'))"`

    # now run Python with the virtualenv set as Python's HOME
    export PYTHONHOME=$ENV
    exec $PYTHON "$@"

You would then execute the python scripts using this executable, e.g.

    $HOME/ENV/bin/frameworkpython autoencoder.py

## Quick start
1. Install the pre-requisites listed above
2. Obtain some input data. The current set-up can read the file in `/afs/cern.ch/work/j/jcatmore/public/dqews/test_ntuple.root`.
3. To run the autoencoder: `$HOME/ENV/bin/python autoencoder.py`
4. To run the classifier: `$HOME/ENV/bin/python classifier.py`

In both cases the training will run, followed by testing. The results will be displayed. Expect each job to take around 5 minutes. In both cases the network structure and weights after training will be recorded to an H5 file, so after you have run once you can adjust the flag in the two scripts called `runTraining` to false, and then the job will just run from the H5 file and skip the training. Obviously if you adjust the neural network structure, you will have to re-train.

## Input data
The input variable names are written in the file called `featuresLists.py`. The assumption is that the input is a ROOT TTree, with the branch names as defined in the featuresLists file. Switching to a different n-tuple file is just a case of changing the names in the featureList file. Switching to direct xAOD input will require a bit more work. **TO DO**: set up xAOD conversion to NumPy.  

Some test files are available, produced from a variety of express stream datasets which are described [here](https://indico.cern.ch/event/630665/contributions/2605145/attachments/1473556/2281167/catmore.pdf) (slide 10 - the same slides show some results obtained from this code).

Location: `/afs/cern.ch/work/j/jcatmore/dqews`
Files: 
* `304008_lb0775-lb0784.root` : main reference
* `304008_lb0785-lb0794.root` : second reference
* `296939.root` : Queen Bork		
* `299241.root`	: toroid off	
* `296942.root`	: pixel HV scan	
* `299278.root`	: King Bork	
* `299184.root`	: pre-d0 bias	
* `299584.root`	: pixel layer 2 out of synch	

The input file names for the reference and the subject are set within each script.

## Description of the auto-encoder example
The auto-encoder example demonstrates the use of an auto-encoder to distinguish between two sample types. An auto-encoder is a network that is trained on its own input, so it learns to an identity transformation for the distribution represented by the training sample. When confronted with data sampled from a different underlying distribution, the weights encoding the identity may no longer be appropriate, leading to the output being very different from the input. Such networks therefore have two advantageous properties for anomaly detection - they only need to be trained on the reference sample, and when a new sample is being tested the measure of how anomalous the sample is can be measured by difference between the input and the output - the *reconstruction error*. 

In the example the script reads in 100K reference objects to use for training, and then separately reads in a second batch of 100K reference objects, and 100K subject objects, for testing. The network itself consists of 71 inputs (corresponding to the 71 variables in the test n-tuples), a hidden layer with 20 units using a rectified linear activation, followed by a linear output layer with 71 outputs (so the network has an "hour glass" shape). 

Once trained, testing procedes by running the network over the training sample, the second unseen reference sample, and the subject sample. The distributions of the reconstruction errors thus obtained are drawn, as is a ROC curve produced by scanning over the reconstruction error distribution for the unseen data. A variable importance map is also produced, showing how much each variable contributes to the overall reconstruction error. 

When used in DQEWS, it is envisaged that the autoencoder would be trained on the reference lumi-block, and then exposed to the one to be tested. It is expected that the reconstruction errors should be small - if they are not, it is a sign of anomalous collective features in the new LB.

## Description of the classifier
The classifier example is very straightforward and consists of a neural network with one hidden layer using a rectified linear activation and sigmoid output layer. The network is fed 100K subject and 100K reference objects for training, and is then tested on a second equal batch. It produces a probability of each event being a signal event, which is then converted into a binary decision. The probabilities are plotted, and the binary decision is used to make the ROC curve. Using a method described in Ecol. Modelling 160 (2003) 249-264, it calculates the relative importance of each variable in the overall separation of the objects.

When used in DQEWS, it is envisaged that the reference lumi block will be treated as the background, and the new LB as the signal. They will be partitioned as in the example, and the network trained and tested on the partitions. If it makes any headway in separating them (e.g. if the ROC area is significantly different from 0.5) this is a clear sign that the two are incompatible.

## Description of the BDT
The BDT is set up in exactly the same way as the classifier. The BDT used is the implementation in SciKitLearn, not the traditional TMVA BDT.


