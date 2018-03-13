from __future__ import print_function

import os
import argparse
import time

import torch
import torchvision
import numpy as np
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs= []
        for name, m in self.submodule._modules.items():
            if name == 'fc':
                x = x.view(1,-1)   

            x = m(x)
#            print(name)
#            print(x)
            if name in self.extracted_layers:
               outputs += [x]
        return outputs


class FinetuneModel(nn.Module):
	def __init__(self, pretrained_model, ncls):
		super(FinetuneModel, self).__init__()
		self.pretrained_model = pretrained_model
		num_fts = self.pretrained_model.fc.in_features
		self.pretrained_model.fc = nn.Linear(num_fts, 2048) 
		self.fc1 = nn.Linear(2048, 1024)
		self.fc2 = nn.Linear(1024, ncls)
		
	def forward(self, x):
		x = self.pretrained_model(x)
		x = self.fc1(x)
		x = self.fc2(x)

		return x

