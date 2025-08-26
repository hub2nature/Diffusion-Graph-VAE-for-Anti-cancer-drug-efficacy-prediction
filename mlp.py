from torch import nn
import numpy as np
import torch
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import DataLoader
from latent import LatentDataset

import torch
import torch.nn as nn



# MLP model
class MLP_simple(nn.Module):

    def __init__(self):
        super(MLP_simple, self).__init__()

        activateLayer = nn.PReLU()

        self.geMLP = nn.Sequential(
            nn.Linear(256, 64),
              
            nn.PReLU(), #change ReLU

            nn.Linear(64, 32),
              
            nn.PReLU(),

            nn.Linear(32, 16)
              
            
            )  # nn.BatchNorm1d(256),


        self.dMLP = nn.Sequential(
            nn.Linear(32, 64),
              
            nn.PReLU(), #change ReLU else explain why using PReLU

            nn.Linear(64, 32),
              
            nn.PReLU(),

            nn.Linear(32, 8),
              
            nn.PReLU())  # nn.BatchNorm1d(128),


        self.combineMLP = nn.Sequential(
            nn.Linear(24, 16),
              
            nn.PReLU(), #change ReLU

            nn.Linear(16, 16),
              
            nn.PReLU(),

            nn.Linear(16, 8),
              
            nn.PReLU(),
            
            nn.Linear(8, 1),
            # nn.PReLU(),
        )  # nn.BatchNorm1d(128),


    def forward(self, geLatentVec, dLatentVec):
        ge = self.geMLP(geLatentVec)
        d = self.dMLP(dLatentVec)

        combination = torch.cat([ge, d], dim=1)
        res = self.combineMLP(combination)

        return res