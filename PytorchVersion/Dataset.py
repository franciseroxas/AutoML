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
        currLayers = getLayerNames(df.get('arch_and_hp')[i], removeParams = False)
        allPossibleNames.update(currLayers)
        allDataInputs.append(currLayers)

        maxLayerLength = max(maxLayerLength, len(currLayers))

    word_to_ix = {word: i for i, word in enumerate(allPossibleNames)}
    return allDataInputs, word_to_ix, maxLayerLength
    

#Get the vocab size
def getVocabSize(inputFn = None, df = None):
    if(df is None):
        df = pd.read_csv(inputFn)
    
    allDataInputs = []
    allPossibleNames = set() 
    maxLayerLength = 0

    for i in range(df.get('arch_and_hp').shape[0]):
        currLayers = getLayerNames(df.get('arch_and_hp')[i], removeParams = False)
        allPossibleNames.update(currLayers)

    return len(allPossibleNames) #for -1

def getModelInputs(df):
    allDataInputs, word_to_ix, maxLayerLength = turnLayersIntoTorchVectorsWithParams(df)

    #turn all of the model layer inputs to tensors 
    maxLayerLength = maxLayerLength + 1
    for i in range(len(allDataInputs)):
        context = allDataInputs[i]
        inputTensor = make_context_vector(context, word_to_ix)
        inputTensor = F.pad(inputTensor, (0, maxLayerLength - inputTensor.shape[0]), "constant", -1)

        allDataInputs[i] = inputTensor

    return torch.tensor(np.array(allDataInputs))
    
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
    
def getModelLosses(df):
    trainLossColumns = np.zeros((df.shape[0], 50))
    valLossColumns = np.zeros((df.shape[0], 50))
    
    for i in range(50):
        trainLossColumns[:,i] = df['train_losses_' + str(i)].values
        valLossColumns[:, i] = df['val_losses_' + str(i)].values
        
    trainLossColumns = torch.unsqueeze(torch.tensor(trainLossColumns), 2)
    valLossColumns = torch.unsqueeze(torch.tensor(valLossColumns), 2)
    lossColumns = torch.cat([trainLossColumns, valLossColumns], 2)
    lossColumns = torch.tensor(lossColumns, dtype = torch.float32)
    return lossColumns    
    
class ModelLayerDataset(Dataset):
    def __init__(self, allModelLayersTensor, lossHistory, outputPredictors):   
        self.allModelLayersTensor = allModelLayersTensor
        self.outputPredictors = outputPredictors
        self.lossHistory = lossHistory

    def __len__(self):
        return len(self.allModelLayersTensor)
        
    def getInputDimLen(self):
        return self.allModelLayersTensor[0].shape[0]

    def __getitem__(self, idx):
        return [self.allModelLayersTensor[idx], self.lossHistory[idx]], self.outputPredictors[idx]
        
def createDataset(inputFn = 'train.csv'):
    df = pd.read_csv(inputFn)
    df = pd.DataFrame(df.dropna().values, columns=df.columns)
    
    allModelLayersTensor = getModelInputs(df) / getVocabSize(df = df)
    outputPredictors = getModelRegressionOutputs(df)
    lossHistory = getModelLosses(df)

    allIndices = np.arange(allModelLayersTensor.shape[0])
    allIndices = np.random.permutation(allIndices)
    trainIndices = allIndices[0:int(allIndices.shape[0] * 0.7)]
    valIndices = allIndices[int(allIndices.shape[0] * 0.7)+1::]
    
    #Get params for the train data loader
    trainModelLayersTensor = allModelLayersTensor[trainIndices]
    trainModelPredictors = outputPredictors[trainIndices]
    trainModelLossHistory = lossHistory[trainIndices]
    
    #Get params for the validation data loader
    valModelLayersTensor = allModelLayersTensor[valIndices]
    valModelPredictors = outputPredictors[valIndices]
    valModelLossHistory = lossHistory[valIndices]
    
    trainingDataset = ModelLayerDataset(trainModelLayersTensor, trainModelLossHistory, trainModelPredictors)
    valDataset = ModelLayerDataset(valModelLayersTensor, valModelLossHistory, valModelPredictors)
    return trainingDataset, valDataset