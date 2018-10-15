## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
                
        #input hxw = 224
        self.convnet = nn.Sequential(
            #conv1
            nn.Conv2d(1,32,kernel_size=5,stride=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(p=0.1),#out = 110
            #conv2
            nn.Conv2d(32,64,kernel_size=3,stride=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(p=0.2),#out= 54
            #conv3
            nn.Conv2d(64,128,kernel_size=3,stride=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(p=0.3),#out=26
            #conv4
            nn.Conv2d(128,256,kernel_size=2,stride=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(p=0.4),#out=12
            nn.AvgPool2d(1)
        )
        
        
        self.densenet = nn.Sequential(
            #dense1
            #nn.Linear(256*12*12,1000),
            #nn.ELU(),
            #nn.Dropout(p=0.5) ,
            #dense2
            #nn.Linear(1000,1000),
            #nn.ELU(),
            #nn.Dropout(p=0.6),
            #dense3
            #nn.Linear(1000,2*68),
            nn.Linear(36864,2*68)
        )

      
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.convnet(x)
        x = x.view(x.size(0), -1)        
        x = self.densenet(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
