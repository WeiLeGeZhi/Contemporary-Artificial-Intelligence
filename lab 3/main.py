from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from models import LeNet, AlexNet, ResNet, VGG16, GoogleNet
from train_and_test import train, acc_rate
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Command")
    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--model', default="LeNet", type=str)
    parser.add_argument('--optimizer', default="SGD", type=str)
    parser.add_argument('--lr_decay_gamma', default=1.0, type=float)
    
    args = parser.parse_args()
    
    batch_size = args.batch_size
    learning_rate = args.lr
    epochs = args.epoch
    
    if args.model == 'LeNet':
        model = LeNet.LeNet()
        transform = transforms.Compose([transforms.ToTensor()])
    elif args.model == 'AlexNet':
        model = AlexNet.AlexNet()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    elif args.model == 'ResNet':
        model = ResNet.ResNet()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    elif args.model == 'VGG':
        model = VGG16.VGG16()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    elif args.model == 'GoogleNet':
        model = GoogleNet.GoogleNet()
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.15, 0.30)])
    else:
        raise ValueError('Unknown model')

    dataset = MNIST(root="./data", train=True, transform=transform, download=True)

    total_size = len(dataset)
    train_size = int((5/6) * total_size) 
    valid_size = total_size - train_size 
    
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = MNIST(root="./data", train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Unknown optimizer")

    schedular = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_gamma)

    result_model = train.train(epochs, model, train_loader, optimizer, valid_loader, schedular)
    accuracy = acc_rate.test(result_model, test_loader)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')