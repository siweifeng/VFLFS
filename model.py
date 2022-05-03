import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

    
class LocalEncoder(nn.Module):
    
    def __init__(self, input_size, embed_size):
        super(LocalEncoder, self).__init__()
        
        self.fc = nn.Linear(input_size, embed_size)
        
    def forward(self, x):
        
        x = F.relu(self.fc(x))
        
        return x

    
class LocalDecoder(nn.Module):
    
    def __init__(self, embed_size, input_size):
        super(LocalDecoder, self).__init__()
        
        self.fc = nn.Linear(embed_size, input_size)
        
    def forward(self, x):
        
        x = F.relu(self.fc(x))
        
        return x
   
    
class Classifier(nn.Module):
    
    def __init__(self, embed_size, n_class):
        super(Classifier, self).__init__()
        
        self.fc = nn.Linear(embed_size, n_class)
        
    def forward(self, x):
        
        x = self.fc(x)
        
        return x