import math
import random

import numpy as np
import torch
import torch.nn as nn


class ArticulationPredictor(nn.Module):
    
    def __init__(self, size_dataset, num_bones, degree):
        super(ArticulationPredictor, self).__init__()
        
        self.degree = degree
        # degree = 10 #(0.17 rad)
        rad = self.degree * (math.pi/180)
        
        # Bone rotations are represented as quaternions, which have 4 components.
        # Using nn.Embedding to act as a lookup table
        self.bones_rotations = nn.Embedding(size_dataset, num_bones * 4) # Shape is torch.Size([49, 60]) 
        # Resizing the bones_rotations to the shape (size_dataset, num_bones, 4)
        self.bones_rotations.weight.data = self.bones_rotations.weight.data.view(size_dataset, num_bones, 4)
        # Initialize with small values to prevent NaNs in quaternion calculations
        self.bones_rotations.weight.data *= 1e-6 
        # Set the first component of each quaternion to 1 for the identity quaternion
        self.bones_rotations.weight.data[:, :, 0] = 1.0
        # Resize back
        self.bones_rotations.weight.data = self.bones_rotations.weight.data.view(size_dataset, num_bones * 4)

        self.num_bones = num_bones
        
        # Initializing the name to index dictionary
        self.name_to_index = {}
        
        
        # Initialize the embedding weights to zeros
        # self.bones_rotations.weight.data.zero_()
        
        # Initialising the bone rotation parameter with zeros to make sure it starts from rest pose.
        # nn.init.zeros_(self.bones_rotations.weight)
        
        # nn.init.uniform_(self.bones_rotations.weight, -rad, rad)
        

            
    def get_sample_index(self, names, device):
        # Sort names for consistent ordering
        names = sorted(names)
        
        indices = []
        for name in names:
            # If the name is not in the dictionary, assign it the next available index
            if name not in self.name_to_index:
                self.name_to_index[name] = len(self.name_to_index)
            indices.append(self.name_to_index[name])

        return torch.tensor(indices, device=device)


    def forward(self, batch):
        # Extracting index of the samples from the batch using the sample names
        
        # degrees to radians
        rad = self.degree * (math.pi/180)
        
        sample_names = batch['name']
        
        sample_index = self.get_sample_index(sample_names, batch['image'].device)

        # The embedding layer processes the batch of indices and retrieve the corresponding 
        # bones_rotations values for each sample in the batch
        bones_rotations = self.bones_rotations(sample_index)
        bones_rotations = bones_rotations.view(-1, self.num_bones, 4)
        
        # # Applying tanh to ensure the values are between -1 and 1
        # bones_rotations = torch.tanh(bones_rotations) * rad

        # Normalize the quaternion
        bones_rotations = nn.functional.normalize(bones_rotations, p=2, dim=2)

        return bones_rotations
