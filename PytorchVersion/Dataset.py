import numpy as np
import csv
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

def find_nth(haystack: str, needle: str, n: int) -> int:
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def helperGetParenthesesIndices(searchString):
    i = 0 
    retArr = []
    stack = []
    while(i < len(searchString)):
        if(searchString[i] == "("):
            stack.append(i)
        elif(searchString[i] == ")"):
            retArr.append(np.array([stack.pop(), i]))
        i = i + 1

    return np.array(retArr)

def getLayerNames(modelArch, removeParams = False):
    layerNames = []
    indicesToGrab = helperGetParenthesesIndices(modelArch)
    
    for indices in indicesToGrab:
        if(indices[0]-1 > 0):
            leftIndice = indices[0]-1
            while(modelArch[leftIndice].isalnum()):
                leftIndice = leftIndice - 1

            leftIndice = leftIndice + 1
            
            if(leftIndice >= indices[0]-1):
                continue
            
            rightIndice = indices[1]+1
            
            if(removeParams):
                rightIndice = find_nth(haystack = modelArch[leftIndice:rightIndice], needle = "(", n = 1) + leftIndice
                
            layerNames.append(modelArch[leftIndice:rightIndice])

    return layerNames
    
#Without params
def turnLayersIntoTorchVectorsWithParams(df):
    allDataInputs = []
    allPossibleNames = set() 
    maxLayerLength = 0

    for i in range(df.get('arch_and_hp').shape[0]):
        currLayers = getLayerNames(df.get('arch_and_hp')[i], removeParams = True)
        allPossibleNames.update(currLayers)
        allDataInputs.append(currLayers)

        maxLayerLength = max(maxLayerLength, len(currLayers))

    word_to_ix = {word: i for i, word in enumerate(allPossibleNames)}
    return allDataInputs, word_to_ix, maxLayerLength
    
def getModelRegressionOutputs(df):
    modelParamsTensorList = []

    #These should go through an MLP
    for i in range(df.get('arch_and_hp').shape[0]):
        modelParamsList = []
        modelParamsList = modelParamsList + [df.get('val_error')[i]]
        modelParamsList = modelParamsList + [df.get('val_loss')[i]]
        modelParamsList = modelParamsList + [df.get('train_error')[i]]
        modelParamsList = modelParamsList + [df.get('train_loss')[i]]
        
        modelParamsTensorList.append(modelParamsList)
    return torch.Tensor(np.array(modelParamsTensorList))
    
def turnModelParamsIntoTorchVectors(df):
    modelParamsTensorList = []

    #These should go through an MLP
    for i in range(df.get('arch_and_hp').shape[0]):
        modelParamsList = []
        modelParamsList = modelParamsList + [df.get('epochs')[i]]
        modelParamsList = modelParamsList + [df.get('number_parameters')[i]]

        modelParamsTensorList.append(modelParamsList)
        
    #These will go through an lstm + mlp 
    initParamsTensorList = []
    maxInitParamsLength = 0
    for i in range(df.get('init_params_mu').shape[0]):
        
        
        init_params_mu = [float(stat) for stat in df.get('init_params_mu')[i].replace('[', '').replace(']', '').split(',')]
        init_params_std = [float(stat) for stat in df.get('init_params_std')[i].replace('[', '').replace(']', '').split(',')]
        init_params_l2 = [float(stat) for stat in df.get('init_params_l2')[i].replace('[', '').replace(']', '').split(',')]
        
        initParamsList = torch.transpose(torch.stack([torch.Tensor(init_params_mu), 
                                                     torch.Tensor(init_params_std),
                                                     torch.Tensor(init_params_l2)]), 0, 1)
        initParamsTensorList.append(initParamsList)
        
        maxInitParamsLength = max(maxInitParamsLength, len(init_params_mu)) #init params list is a list of variable x 3 Tensors

    return modelParamsTensorList, len(modelParamsList), initParamsTensorList, maxInitParamsLength

#Get the vocab size
def getVocabSize(inputFn):
    df = pd.read_csv(inputFn)
    
    allDataInputs = []
    allPossibleNames = set() 
    maxLayerLength = 0

    for i in range(df.get('arch_and_hp').shape[0]):
        currLayers = getLayerNames(df.get('arch_and_hp')[i], removeParams = True)
        allPossibleNames.update(currLayers)

    return len(allPossibleNames) + 1 #for -1
    
#Without params
def turnLayersIntoTorchVectorsWithoutParams(df):
    allDataInputs = []
    allPossibleNames = set() 
    maxLayerLength = 0

    for i in range(df.get('arch_and_hp').shape[0]):
        currLayers = getLayerNames(df.get('arch_and_hp')[i], removeParams = True)
        allPossibleNames.update(currLayers)
        allDataInputs.append(currLayers)

        maxLayerLength = max(maxLayerLength, len(currLayers))

    word_to_ix = {word: i for i, word in enumerate(allPossibleNames)}
    return allDataInputs, word_to_ix, maxLayerLength

def getModelInputs(df):
    allDataInputs, word_to_ix, maxLayerLength = turnLayersIntoTorchVectorsWithoutParams(df)
    modelParamsTensorList, _, initParamsTensorList, maxInitParamsLength = turnModelParamsIntoTorchVectors(df)

    #turn all of the model layer inputs to tensors 
    maxLayerLength = maxLayerLength + 1
    for i in range(len(allDataInputs)):
        context = allDataInputs[i]
        inputTensor = make_context_vector(context, word_to_ix)
        inputTensor = F.pad(inputTensor, (0, maxLayerLength - inputTensor.shape[0]), "constant", -1)

        allDataInputs[i] = inputTensor
    
    #Second for loop for readability
    maxInitParamsLength = maxInitParamsLength + 1
    for i in range(len(initParamsTensorList)):
        #Pad the 2nd to last dimension to make it the full length
        currInitParams = F.pad(initParamsTensorList[i], (0,0, 0, maxInitParamsLength - initParamsTensorList[i].shape[0]), "constant", 0)
        currInitParams = torch.reshape(currInitParams, (-1, 1)).squeeze(1)
        initParamsTensorList[i] = currInitParams
    
    #Turn to tensors
    allModelLayersTensor = torch.Tensor(np.array(allDataInputs)) #Padded using the value -1 as the end of model statement
    initParamsTensor = torch.Tensor(np.array(initParamsTensorList))
    modelTrainValLossesTensor = torch.Tensor(np.array(df.get(['train_losses_' + str(i) for i in range(50)]).values, dtype = 'float64'))

    return allModelLayersTensor, torch.Tensor(np.array(modelParamsTensorList)), initParamsTensor, modelTrainValLossesTensor
    
class ModelLayerDataset(Dataset):
    def __init__(self, allModelLayersTensor, modelParamsTensor, initParamsTensor, modelTrainValLossesTensor, outputPredictors):        
        self.allModelLayersTensor = allModelLayersTensor
        self.modelParamsTensor = modelParamsTensor
        self.initParamsTensor = initParamsTensor
        self.modelTrainValLossesTensor = modelTrainValLossesTensor
        
        self.outputPredictors = outputPredictors

    def __len__(self):
        return len(self.allModelLayersTensor)
        
    def getInputDimLen(self):
        return self.allModelLayersTensor[0].shape[0]

    def __getitem__(self, idx):
        return [self.allModelLayersTensor[idx], self.modelParamsTensor[idx], self.initParamsTensor[idx], self.modelTrainValLossesTensor[idx]], self.outputPredictors[idx]
        
def createDataset(inputFn = 'train.csv'):
    df = pd.read_csv(inputFn)
    df = pd.DataFrame(df.dropna().values, columns=df.columns)
    allModelLayersTensor, modelParamsTensor, initParamsTensor, modelTrainValLossesTensor = getModelInputs(df)
    outputPredictors = getModelRegressionOutputs(df)

    allIndices = np.arange(allModelLayersTensor.shape[0])
    allIndices = np.random.permutation(allIndices)
    trainIndices = allIndices[0:int(allIndices.shape[0] * 0.7)]
    valIndices = allIndices[int(allIndices.shape[0] * 0.7)+1::]

    #Get params for the train data loader
    trainModelLayersTensor = allModelLayersTensor[trainIndices]
    trainModelParamsTensor = modelParamsTensor[trainIndices]
    trainInitParamsTensor = initParamsTensor[trainIndices]
    trainTrainValLossesTensor = modelTrainValLossesTensor[trainIndices]
    trainPredictors = outputPredictors[trainIndices]

    #Get params for the validation data loader
    valModelLayersTensor = allModelLayersTensor[valIndices]
    valModelParamsTensor = modelParamsTensor[valIndices]
    valInitParamsTensor = initParamsTensor[valIndices]
    valTrainValLossesTensor = modelTrainValLossesTensor[valIndices]
    valPredictors = outputPredictors[valIndices]

    trainingDataset = ModelLayerDataset(trainModelLayersTensor, trainModelParamsTensor, trainInitParamsTensor, trainTrainValLossesTensor, trainPredictors)
    valDataset = ModelLayerDataset(valModelLayersTensor, valModelParamsTensor, valInitParamsTensor, valTrainValLossesTensor, valPredictors)
    return trainingDataset, valDataset