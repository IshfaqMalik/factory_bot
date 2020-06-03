import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import Model
import utils

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', action='store', help='directory containing images')
parser.add_argument('--save_dir', action='store', help='save trained checkpoint to this directory' )
parser.add_argument('--arch', action='store', help='what kind of pretrained architecture to use', default='vgg16')
parser.add_argument('--gpu', action='store_true', help='use gpu to train model')
parser.add_argument('--epochs', action='store', help='# of epochs to train', type=int, default=20)
parser.add_argument('--hidden_units', action='store', help='# of hidden units to add to model', type=int, default=512)
parser.add_argument('--output_units', action='store', help='# of classes to output', type=int, default=102)

args=parser.parse_args()

#Sorting the data for training, validation, testing

data_dir = 'flowers'

train_data, valid_data, test_data, trainloader, validloader, testloader = utils.data_loader(data_dir)


#Create Model 

model = utils.network_model(args.arch)

for param in model.parameters():
    param.requires_grad = False

    
input_units = utils.get_input_units(model, args.arch)

model.classifier = Model.Network(input_units, args.output_units,[args.hidden_units],drop_p=0.5) 
 
 # train  Model 

optimizer = optim.Adam(model.classifier.parameters(), lr=0.01)
criterion = nn.NLLLoss()

Model.train_network(model, trainloader, validloader, optimizer,criterion, args.epochs, args.gpu)

# testing Model 

test_loss, test_accuracy = test_network(model, testloader, criterion, args.gpu)

print("\n ---\n Test: {:.2f} %".format(test_accuracy*100), "Test Loss: {}".format(test_loss))

# save netowrk 

Model.save_checkpoint(model, train_data, optimizer, args.save_dir, args.arch)


