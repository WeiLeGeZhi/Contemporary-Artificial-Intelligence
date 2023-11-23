import torch.nn as nn
from train_and_test import acc_rate


def train(epochs, model, train_loader, optimizer, valid_loader, schedular):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            schedular.step()

        if (epoch + 1) % 1 == 0:
            accuracy = acc_rate.test(model, valid_loader)
            print(f'Epoch {epoch + 1}/{epochs}, Validation Accuracy: {accuracy * 100:.2f}%')
    return model
