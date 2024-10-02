import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def findConv2dOutShape(hin, win, conv, pool=2):
    # Calculate the output dimensions after applying a Conv2D layer
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation

    # Compute the output height and width after the convolution
    hout = np.floor((hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    wout = np.floor((win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

    # Apply pooling if specified
    if pool:
        hout /= pool
        wout /= pool

    return int(hout), int(wout)


class CNN_BT(nn.Module):

    def __init__(self, params):
        super(CNN_BT, self).__init__()

        # Extract parameters from the input dictionary
        Cin, Hin, Win = params["shape_in"]
        init_f = params["initial_filters"]
        num_fc1 = params["num_fc1"]
        num_classes = params["num_classes"]
        self.dropout_rate = params["dropout_rate"]

        # Convolution layers
        self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3)
        h, w = findConv2dOutShape(Hin, Win, self.conv1)

        self.conv2 = nn.Conv2d(init_f, 2 * init_f, kernel_size=3)
        h, w = findConv2dOutShape(h, w, self.conv2)

        self.conv3 = nn.Conv2d(2 * init_f, 4 * init_f, kernel_size=3)
        h, w = findConv2dOutShape(h, w, self.conv3)

        self.conv4 = nn.Conv2d(4 * init_f, 8 * init_f, kernel_size=3)
        h, w = findConv2dOutShape(h, w, self.conv4)

        # Fully connected layers
        self.num_flatten = h * w * 8 * init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def forward(self, X):
        # Forward pass through the network
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2, 2)

        # Flatten and pass through fully connected layers
        X = X.view(-1, self.num_flatten)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, self.dropout_rate)
        X = self.fc2(X)

        # Output probabilities using log softmax
        return F.log_softmax(X, dim=1)


# Example parameters
params = {
    "shape_in": (3, 256, 256),  # Input shape (channels, height, width)
    "initial_filters": 8,        # Initial number of filters
    "num_fc1": 100,              # Number of neurons in the first fully connected layer
    "dropout_rate": 0.25,        # Dropout rate
    "num_classes": 2             # Number of output classes
}

# Instantiate the model
cnn_model = CNN_BT(params)

# Specify device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to device
model = cnn_model.to(device)
