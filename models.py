import torch
import torch.nn as nn
import torch.optim as optim 
import pandas as pd 
import numpy as np

#TODO: cluster y-vectors based on behavour/trend
class KMEANS:
    def __init__(self):
        return None

class NaiveRepeaterModel:
    def __init__(self, output_length=20):
        self.output_length = output_length
        self.last_two = None
    
    def fit(self, X, y=None):
        self.last_two = X[:, -2:]
    
    def predict(self, X):
        #num_samples = X.shape[0]
        repeats = self.output_length // 2
        preds = np.tile(self.last_two, (1, repeats))
        return preds


class MultiOutputRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim) 
    
    def forward(self, x):
        return self.linear(x)
    



#TODO: Gaussian Discriminant Analysis

#TODO: SVM Model

#TODO: Sequence to one transformer model 