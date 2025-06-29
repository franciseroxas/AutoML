# AutoML
Auto Machine Learning Final Project completed with Micheal Shoen.

## Problem Description
The following text is based on the original problem statement (link: https://www.kaggle.com/c/cs446-fa19/overview). 

Selecting model parameters is often a difficult task in applied machine learning. Automated machine learning (AutoML) is a framework of tools designed to automate this model selection process. There is rising evidence in \[1, 2, 3\] that this prediction of performance can be done without running a trained model. 

The general strategy that is emploted is to create a dataset **D** where the training data is a set of model hyperparemeters **X** and the output reression values is a set of associated model performance metrics **Y** of models from **H** on another relevant dataset (Cifar-10 in this case).

This repository exists to document how I and Micheal attempted this process in Tensorflow (and how I subsequently recreated the results in PyTorch).

References:

1. Neural Architecture Optimization, https://arxiv.org/abs/1808.07233

2. Progressive Neural Architecture Search, https://arxiv.org/abs/1712.00559

3. Accelerating Neural Architecture Search using Performance Prediction, https://arxiv.org/abs/1705.10823

## Dataset Description
The text below is based on the dataset description found here: https://www.kaggle.com/c/cs446-fa19/data. 

The dataset **D** contains a collection of textual descriptions of neural network models trained and evaluated on the Cifar-10 dataset. Explicitly, the inputs **X** are model hyperparemeters such as architecture layers used, batch size, criterion, number of epochs, etc ... The output regressors **Y** are the associate model performance metrices of these models evaluated on Cifar-10 using the set of criteria from **X**. The dataset can be found in the file train.csv and a 'hidden' test set can be found in test.csv. 

**Some columns of train.csv:**
The following were used as train data:
- id: identification for the data sample
- arch_and_hp: model architecture and hyperparameters used with that architecture
- batch_size_test: number of samples that are evaluated at once for test data
- batch_size_val: number of samples that are evaluated at once for validation data
- criterion:  loss function
- epochs: number of epochs the model was trained
- number_parameters: number of parameters in the model
- optimizer: optimizer used to update gradients
- batch_size_train: number of samples that are evaluated at once for training data
- init_params_mu: mean of initial parameters
- init_params_std: standard deviation of initial parameters
- init_params_l2: L2 norm of initial parameters

The following were considered regressors:
- val_accs_{0,..,49}: validation accuracy for the first 50 epochs
- val_losses_{0,..,49}: validation losses for the first 50 epochs
- train_accs_{0,..,49}: training accuracy for the first 50 epochs
- train_losses_{0,..,49}: training losses for the first 50 epochs

## About this repo 
In this repository, there is our original submission in the _tensorFlowVersion_ folder and my second attempt at this many years later in the _Pytorch_ Version folder. 

The datasets used are included in the _Datasets_ folder and copies of the problem description in the _problemDescription_ folder. 

## General approach
Both approaches took in the input model parameters (text) and fed them through an embedding layer to get a numeric vector. 

Afterwards, the numeric vector was fed through an Long Short Term Memory unit (LSTM) model and then fed to a simple Multi Layer Perceptron (MLP) to get the output regressors. 

The Pytorch version differs from the Tensorflow version by feeding in "non-variable length" model parameters such as criterion, batch size, num epochs, etc... into a seperate MLP. The output of this MLP is concatenated with the output of the LSTM before being fed into one final MLP to obtain the model regressors. 