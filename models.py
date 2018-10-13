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
        #self.conv1 = nn.Conv2d(1, 32, 5)
        
        #self.layer1 = nn.Sequential(
        #    nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
        #    nn.BatchNorm2d(16),
        #    nn.ReLU(),
        #    nn.MaxPool2d(kernel_size=2, stride=2))
        
        #input hxw = 224
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=5,stride=1),
            nn.Relu(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(p=0.1)
        )
        #out = 110
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=1),
            nn.Relu(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(p=0.2)
        )
        #out= 54
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,stride=1),
            nn.Relu(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(p=0.3)
        )
        #out=26
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128,256,kernel_size=3,stride=1),
            nn.Relu(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(p=0.4)
        )
        #out=12
        self.conv5 = nn.Sequential(
            nn.Conv2d(256,512,kernel_size=3,stride=1),
            nn.Relu(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(p=0.4)
        )
        #out=5
        
        self.dense1 = nn.Sequential(
            nn.Linear(512*5*5,1000),
            nn.Relu(),
            nn.Dropout(p=0.5)           
        )
        
        self.dense2 = nn.Sequential(
            nn.Linear(1000,1000),
            nn.Relu(),
            nn.Dropout(p=0.6)           
        )
        
        self.dense3 = nn.Sequential(
            nn.Linear(1000,2),
        )
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
