import numpy as np 
import torch
from torchvision import transforms, datasets,models
from PIL import Image 

import json


def process_image(image):
    
    
    
    img = Image.open(image_path)
    
    transform = transfoems.Compose([transform.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])
                                  
    
                                  
    transformed_img = transform(img)
                                   
    img = np.array(transformed_img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = (np.transpose(img,(1,2,0)) - mean) / std 
    image = np.transpose(image, (2, 0, 1))
    
    return image
                                   
                            
                                   
def data_loader(data_dir):
                                   
    train_dir = data_dir +'/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
                                   
    data_transforms = transforms.Compose([transforms.Resize(255), 
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],
                                                                  [0.229, 0.224, 0.225])])
                                          
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                            transforms.Resize(255), 
                                            transforms.CenterCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485,0.456,0.406],
                                                                  [0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform= data_transforms)
    test_data = datasets.ImageFolder(test_dir, transform= data_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64)
    
    return train_data, valid_data, test_data, trainloader, validloader, testloader


def resultdisplay(image, probs, classes, top_k):
    #show image
    fig, ax = plt.subplots()
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = np.squeeze(image.numpy(), axis=0).transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    
    #show probabilities bargraph
    fig, ax = plt.subplots()
    ax.barh(np.arange(top_k), probs)
    ax.set_aspect(0.1)
    ax.set_yticks(np.arange(top_k))
    ax.set_yticklabels(classes, size='small')
    ax.set_title('Class Probability')
    ax.set_xlim(0,max(probs)+0.1)
    plt.tight_layout()
    plt.show()
       
    return ax

def get_class(classes, checkpoint, category_names):
    class_to_idx = torch.load(checkpoint)['class_idx'] 
    idx_to_class = {idx: pic for pic, idx in class_to_idx.items()} #geta dict with mapping (class index, class 'name')
    
    if(category_names != None): #take index number, change to class number, and then to flower name
        names = []
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)

        for i in classes:
            category = idx_to_class[i] #convert index of top5 to class number
            name = cat_to_name[category] #convert category/class number to flower name
            names.append(name)
        return names
    
    else: #we just want to take the index number and change it to real class number
        class_id = []
        for i in classes:
            class_id.append(idx_to_class[i])
        return class_id    
    
def show_classes(probs, classes, top_k):
    print('--------Predictions for Image--------')
    i = 0
    while (i < top_k):
        print('%*s. Class: %*s. Pr= %.4f'%(7, i+1, 3, classes[i], probabilities[i]))
        i += 1
                                   
            
def network_model(arch):
     try: 
            model = getattr(models, arch)(pretrained=True)
            return model
     except AttributeError:
         print("%s is not valid torchvision model" % arch)
         raise SystemExit
     else:
        print("error loading model")
        raise SystemExit
    
def get_input_units(model, arch):
    input_size = 0
    
    if('vgg' in arch): return model.classifier[0].in_features
    elif('densenet' in arch): return model.classifier.in_features
    elif('squeezenet' in arch): return model.classifier[1].in_channels
    elif(('resnet' in arch) or ('inception'in arch) ): return model.fc.in_features
    elif('alexnet' in arch): return model.classifier[1].in_features
        
    if(input_size == 0): raise Error    
    return input_units    
    
    