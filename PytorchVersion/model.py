import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
      
#For model layers only    
class LSTMWithMLP(nn.Module):
    def __init__(self, sequence_length, model_size, context_size = 1):
        super(LSTMWithMLP, self).__init__()
        self.lstm1 = nn.LSTM(input_size = 2, hidden_size = 32, num_layers=16, batch_first=True)
        self.mlpForModel = nn.Sequential(nn.Linear(model_size, 32),
                                         nn.ReLU(),
                                         nn.LayerNorm(32),
                                         nn.Linear(32, 32),
                                         nn.ReLU(),
                                         nn.LayerNorm(32),
                                         nn.Linear(32, 32),
                                         nn.ReLU(),
                                         nn.LayerNorm(32))
        
        self.linear1 = nn.Linear(32, 32)
        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.linear2 = nn.Linear(32, 32)
        self.relu2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm1d(32)
        
        self.linearFinal = nn.Linear((sequence_length + 1)*32, 4)

    def forward(self, inputSequence, inputModel):

        outputSequence, (h_n, c_n) = self.lstm1(inputSequence)
        outputModel = self.mlpForModel(inputModel)
        outputModel = torch.unsqueeze(outputModel, 1)
        
        output = torch.cat([outputModel, outputSequence], 1)    
        x = self.linear1(output)
        x = self.relu1(x)
        x = self.batchnorm1(torch.transpose(x, 1, 2))
        x = torch.transpose(x, 1, 2)
        
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.batchnorm2(torch.transpose(x, 1, 2))
        x = torch.transpose(x, 1, 2)
        x = torch.reshape(x, (x.shape[0],-1,))
        
        out = self.linearFinal(x)
        return out
        
       