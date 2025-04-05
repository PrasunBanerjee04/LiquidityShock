import torch
import torch.nn as nn
import torch.optim as optim 
import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC

from models import KMEANS, NaiveRepeaterModel, MultiOutputRegression, GaussianDiscriminantAnalysis, SVMClassifier


class Load:
# load data and get into a pandas dataframe 

class Transform: 
# clean, remove all unwanted columns 
# cluster output values and augment the dataframe 

class Trainer:
# train a selected model on the cleaned data 


class Evaluator: 
# evaluate MSE of the model 