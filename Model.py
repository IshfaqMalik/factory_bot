import torch



from torch import nn, optim

from torchvision import models

import torch.nn.functional as F
import time

import numpy as np

class Network(nn.Module):
    def __init__(self, input_units, output_units, hidden_units, drop_p):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_units, hidden_units[0])])

        #create hidden layers
        layer_sizes = zip(hidden_units[:-1], hidden_units[1:]) #gives input/output units for each layer
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_units[-1], output_units)
        self.dropout = nn.Dropout(p=drop_p)
    
    def forward(self, x):
        for each in self.hidden_layers:
            x = F.relu(each(x)) #apply relu to each hidden node
            x = self.dropout(x) #apply dropout
        x = self.output(x) #apply output weights
        return F.log_softmax(x, dim=1) 
    
    
def train_network(model, trainloader, validloader,optimizer,criterion, epochs, gpu):
    
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    
    
    
    model.to(device)
    
    training_loss, valid_loss = 0, 0
    valid_accuracy = 0
    
    validation_step = True
    
    
    
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(),
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            training_loss = training_loss + loss.item()
            
            #checking validation 
            
            if validation_step == True:
                model.eval()
                
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps,labels)
                        valid_loss = valid_loss + batch_loss.item()
                        
                        #checking accuracy
                        probs = torch.exp(logps)
                        top_p, top_class = probs.topk(1, dim = 1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy = valid_accuracy + torch.mean(equals.type(torch.FloatTensor)).item()
                        
    
    print('\nEpoch: {}/{} '.format(epoch + 1, epochs),
          '\n    Training:\n      Loss: {:.4f}  '.format(training_loss / len(trainloader)))
    
    print("\n    Validation:\n      Loss: {:.4f}  ".format(valid_loss / len(validloader)),
          "Accuracy: {:.4f}".format(valid_accuracy / len(validloader)))
    
    
    model.train()
   


def save_checkpoint(model,train_data,optimizer, save_dir,arch):
    
    
    model_checkpoint = {'arch':arch,
                        
                        'input_size': input_features,
                        'output_size': output_units,
                        'learning_rate': 0.01,       
                        'batch_size': 64,
                        'classifier' : model.classifier,
                        'optimizer': optimizer.state_dict(),
                        'state_dict': model.state_dict(),
                        'class_to_idx': train_data.class_to_idx,}
    if(save_dir == None): torch.save(model_checkpoint, 'checkpoint.pth')
    else:torch.save(model_checkpoint, save_dir+'checkpoint.pth')
        
        
def load_model (file_path, gpu):
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: 
        device = "cpu"
    checkpoint = torch.load(file_path)
    learning_rate = checkpoint['learning_rate']
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def predict(image_path, model, topk):
  
    cuda = torch.cuda.is_available()
    if cuda:
        # Move model parameters to the GPU
        model.cuda()
        print("Number of GPUs:", torch.cuda.device_count())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.device_count()-1))
    else:
        model.cpu()
        print("We go for CPU")
    
    # turn off dropout
    model.eval()

    
    
    
    # tranfer to tensor
    image = torch.from_numpy(np.array([image_path])).float()
    
    # The image becomes the input
    
    if cuda:
        image = image.cuda()
        
    output = model.forward(image)
    
    probabilities = torch.exp(output).data
    
    # getting the topk (=5) probabilites and indexes
    # 0 -> probabilities
    # 1 -> index
    prob = torch.topk(probabilities, topk)[0].tolist()[0] # probabilities
    index = torch.topk(probabilities, topk)[1].tolist()[0] # index
    
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])

    # transfer index to label
    label = []
    for i in range(5):
        label.append(ind[index[i]])

    return prob, label
    
    
 #Testing 

def test_network (model,loader, criterion,gpu):
        if gpu:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
         device = "cpu"
        
        test_loss = 0
        test_accuracy = 0
        
        model.eval()
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            loss = criterion(logps,labels)
            test_loss = test_loss + loss.item()
    
            probabilities = torch.exp(logps)
            top_p, top_class = probabilities.topk(1, dim =1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy = test_accuracy + torch.mean(equals.type(torch.FloatTensor)).item()
            
            return test_loss/len(loader), test_accuracy/len(loader)





                        
                        
                        
                
            
   
    
        
    
        
        
    