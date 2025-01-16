from Dataset import createDataset, getVocabSize
from model import LSTMWithMLP
from torch.utils.data import DataLoader

import numpy as np
import argparse
import torch
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def train(nEpochs = 100, batch_size = 64, inputFn = 'train.csv'):    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Load training and validation sets
    trainingDataset, valDataset = createDataset(inputFn = inputFn)
    train_dataloader = DataLoader(trainingDataset, batch_size=batch_size, shuffle=True)
    
    #Load the models
    model = LSTMWithMLP(sequence_length = 50, model_size = trainingDataset.getInputDimLen())
    if torch.cuda.is_available():
        model.cuda()
    
    learningRate = 0.00001*np.sqrt(batch_size)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, betas=(0.9, 0.999), eps=1e-08)

    all_loss = []
    #Training loop 
    for epoch in range(nEpochs):
        start = time.time()
        currentLoss = 0
        for batch_ndx, data in enumerate(train_dataloader):
            modelParams, outputPredictor = data
            allModelLayersTensor = modelParams[0].to(device)
            lossHistory = modelParams[1].to(device)
            outputPredictor = outputPredictor.to(device)
            
            predictedModelLoss = model(inputSequence = lossHistory, inputModel = allModelLayersTensor)
            loss = criterion(predictedModelLoss, outputPredictor)
            batchLoss = loss.cpu().detach().numpy() / batch_size
            currentLoss = currentLoss + loss.cpu().detach().numpy()
            
            print("Epoch", epoch, " | Batch", batch_ndx, " | Total", len(train_dataloader)," | Batch Loss:", batchLoss)
            print("Truth", outputPredictor[0].cpu().detach().numpy(), "AutoML", predictedModelLoss[0].cpu().detach().numpy())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        all_loss.append(currentLoss)
        
        plt.figure()
        plt.title("Loss vs. Epoch")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.plot(all_loss)
        plt.savefig("Loss_vs_epoch.png")
        
    torch.save({
            'valDataset': valDataset,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'learningRate': learningRate,
            'batch_size': batch_size,
            }, "./AutoML_Model.pt")
    return 
            
def validate(batch_size = 32, inputFn = 'train.csv'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    modelDict = torch.load('AutoML_Model.pt')
    _, valDataset = createDataset(inputFn = inputFn)
    
    val_dataloader = DataLoader(valDataset, batch_size=batch_size, shuffle=True)
    
    model = LSTMWithMLP(sequence_length = 50, model_size = valDataset.getInputDimLen())
    model.load_state_dict(modelDict['model_state_dict'])
    #Validation
    model.eval()
    
    if torch.cuda.is_available():
        model.cuda()
    
    print("Evaluating the model")
    with torch.no_grad():
        for batch_ndx, data in enumerate(val_dataloader):
            modelParams, outputPredictor = data
            allModelLayersTensor = modelParams[0].to(device)
            lossHistory = modelParams[1].to(device)
            outputPredictor = outputPredictor.to(device)
            
            predictedModelLoss = model(inputSequence = lossHistory, inputModel = allModelLayersTensor)
            print("Truth", outputPredictor[0].cpu().detach().numpy(), "AutoML", predictedModelLoss[0].cpu().detach().numpy())
    return 
    
def main(args = None):   
    parser = argparse.ArgumentParser(description='AutoML Project in Pytorch')
    parser.add_argument('--nEpochs', type = int, help = "Number of Epochs to train the model", default = 100)
    parser.add_argument('--batchSize', type = int, help = "Batch size to train the model", default = 64)

    args = parser.parse_args()
    
    #Train the model
    train(nEpochs = args.nEpochs, batch_size = args.batchSize)
    #Validate the model
    validate()
    return

if __name__ == "__main__":
    main()
