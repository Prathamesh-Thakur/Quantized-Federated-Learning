"""pytorchexample: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from medmnist import OCTMNIST


class Net(nn.Module):
    """Define the model to be used. Loaded model is ResNet18."""

    def __init__(self):
        super(Net, self).__init__()
        
        # Load the model from pytorch's model library
        self.model = models.resnet18(weights = "ResNet18_Weights.DEFAULT")
        
        # Modify the first layer to take in a single input channel since OCTMNIST images are grayscale
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        
        # Original number of output features of the model
        num_features = self.model.fc.in_features

        # Add a new linear layer to map it to the 4 output classes of OCTMNIST
        self.model.fc = nn.Linear(num_features, 4)

    def forward(self, x):
        # Run the model on the input
        return self.model(x) 

# Pytorch transforms to be used. Resizing from 28x28 to 224x224 for ReseNet18, standardize and normalize.
pytorch_transforms = Compose([Resize(224), ToTensor(), Normalize([0.5], [0.5])])


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition OCTMNIST data"""
    # Load the training split of OCTMNIST (consists of 100,000+ images)
    train_dataset = OCTMNIST(split = "train", download = True, transform = pytorch_transforms)

    # Calculate the size of a single partition
    partition_size = len(train_dataset) // num_partitions
    
    # Create a list of partition size for random_split
    lengths = [partition_size] * num_partitions

    # Add whatever length remains (in case num_partitions doesn't fully divide the length) to the last partition
    lengths[-1] += len(train_dataset) - sum(lengths)

    # Randomly split the dataset into num_partitions partitions
    partitions = random_split(train_dataset, lengths, generator = torch.Generator().manual_seed(42))
    
    # Grab the subset corresponding to the partition_id
    client_dataset = partitions[partition_id]

    # Calculate lengths for training and validation sets, and perform random split again
    val_len = int(len(client_dataset) * 0.2)
    train_len = len(client_dataset) - val_len
    train_sub, val_sub = random_split(client_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(1234))

    # Initialize training and testing dataloader objects
    trainloader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(val_sub, batch_size=batch_size)

    return trainloader, testloader


def load_centralized_dataset():
    """Load global(server) test set and return dataloader."""
    # Load entire test set
    dataset = OCTMNIST(split="test", download=True, transform = pytorch_transforms)
    
    # Return dataloader object
    return DataLoader(dataset, batch_size=128)


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available

    # Define CrossEntropy loss function.
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    # Use SGD as optimizer for clients
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    
    # Maintain global running loss across epochs
    running_loss = 0.0

    for _ in range(epochs):
        for images, labels in trainloader:
            # Transfer the images to device if available
            images = images.to(device)

            # Squeeze the label tensor for CrossEntropy and transfer to device
            labels = labels.squeeze().long().to(device)

            # Set optimizer to zero
            optimizer.zero_grad()

            # Calculate loss
            loss = criterion(net(images), labels)
            
            # Backpropogation
            loss.backward()
            optimizer.step()

            # Add current epoch loss to total loss
            running_loss += loss.item()
    
    # Calculate average loss across all epochs and data
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device) # Move model to GPU if available

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    
    # Set to eval
    with torch.no_grad():
        for images, labels in testloader:
            # Transfer the images to device if available
            images = images.to(device)

            # Squeeze the label tensor for CrossEntropy and transfer to device
            labels = labels.squeeze().long().to(device)
            
            # Get the outputs
            outputs = net(images)

            # Calculate the loss
            loss += criterion(outputs, labels).item()
            
            # Check how many outputs are correct
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    
    # Calculate accuracy and loss
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    
    return loss, accuracy
