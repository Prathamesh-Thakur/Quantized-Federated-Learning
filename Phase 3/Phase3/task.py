"""pytorchexample: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from medmnist import OCTMNIST


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.model = models.resnet18(weights = "ResNet18_Weights.DEFAULT")
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 4)

    def forward(self, x):
        return self.model(x) 

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
    """Load test set and return dataloader."""
    # Load entire test set
    dataset = OCTMNIST(split="test", download=True, transform = pytorch_transforms)
    return DataLoader(dataset, batch_size=128)


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
