"""Trains the model used for the prediction"""
import torch
import torch.optim as optim
import torch.utils.data
# import torch.backends.cudnn as cudnn
# import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
import math
# # import pandas as pd
from PIL import Image
import random as ra
import shutil
import os
import string
from my_constants import *


class CNN(nn.Module):
    """Class for the CNN."""

    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        self.flatten = nn.Flatten()
        self.conv = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 20, (5, 5)),
            # nn.BatchNorm2d(20),
            nn.ReLU(),
            # nn.Tanh(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Conv2d(20, 50, (5, 5)),
            # nn.BatchNorm2d(50),
            nn.ReLU(),
            # nn.Tanh(),
            nn.MaxPool2d((2, 2), stride=(2, 2))
            # Added
            # nn.Conv2d(50, 50, (5,5)),
            # nn.ReLU(),
            # nn.MaxPool2d((2,2), stride = (2,2))
        )
        self.lin = nn.Sequential(
            nn.Linear(4*4*50, 500),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(500, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.lin(x)
        return x


def split_to_folders() -> None:
    "Splits the images to train, dev, test folders"
    alphabets = list(string.ascii_uppercase)
    alphabet_paths = [IMAGES_DIR + alphabet + "/" for alphabet in alphabets]
    paths = [TRAIN_DIR, TEST_DIR, DEV_DIR]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
        for alphabet in alphabets:
            if not os.path.exists(path + alphabet + "/"):
                os.makedirs(path + alphabet + "/")
    for alphabet_path in alphabet_paths:
        alphabet = alphabet_path.split("/")[-2]
        image_file_names = os.listdir(alphabet_path)
        for image_file_name in image_file_names:
            image_full_path = alphabet_path + image_file_name
            if os.path.isfile(image_full_path):
                division = ra.randint(1, 4)
                if division in (1, 2):
                    shutil.copyfile(image_full_path, TRAIN_DIR +
                                    alphabet + "/" + image_file_name)
                if division == 3:
                    shutil.copyfile(image_full_path, DEV_DIR +
                                    alphabet + "/" + image_file_name)
                if division == 4:
                    shutil.copyfile(image_full_path, TEST_DIR +
                                    alphabet + "/" + image_file_name)


def create_loaders() -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader,
                              torch.utils.data.DataLoader]:
    """Creates data loaders for train, test, and dev."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()])

    train_transform = transform
    test_transform = transform

    train_set = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    dev_set = datasets.ImageFolder(DEV_DIR,   transform=test_transform)
    test_set = datasets.ImageFolder(TEST_DIR,  transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=BATCH_SIZE_TEST, shuffle=False)
    dev_loader = torch.utils.data.DataLoader(dataset=dev_set, shuffle=False)
    return train_loader, dev_loader, test_loader


def training(model: CNN, loss_function: nn.NLLLoss, optimizer: optim.Adam, device: torch.device,
             train_loader: torch.utils.data.DataLoader, dev_loader: torch.utils.data.DataLoader) -> None:
    """"Training loop for the CNN."""
    dev_loss = math.inf
    dev_losses = []
    dev_accuracies = []
    stop_early = False

    for epoch in range(N_EPOCHS):
        if stop_early:
            break
        train_loss = 0
        train_correct = 0
        total = 0
        for batch_num, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            pred = model(data)
            loss = loss_function(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += len(data)
            train_loss += loss.item()
            train_correct += (pred.argmax(1) ==
                              target).type(torch.float).sum().item()

            print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' %
                  (epoch+1, batch_num+1, len(train_loader), train_loss / (batch_num + 1),
                   100. * train_correct / total, train_correct, total))

        cur_dev_loss = 0
        dev_correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(dev_loader):
                data, target = data.to(device), target.to(device)
                pred = model(data)
                loss = loss_function(pred, target)

                total += len(data)
                cur_dev_loss += loss.item()
                dev_correct += (pred.argmax(1) ==
                                target).type(torch.float).sum().item()

            current_loss = cur_dev_loss / (len(dev_loader) + 1)
            dev_losses.append(current_loss)
            current_accuracy = 100. * dev_correct / total
            dev_accuracies.append(current_accuracy)

            if current_loss <= dev_loss:
                dev_loss = current_loss
            else:
                stop_early = True

            print('Evaluating: Batch %d/%d: Loss: %.4f | Dev Acc: %.3f%% (%d/%d)' %
                  (batch_num+1, len(dev_loader), cur_dev_loss / (len(dev_loader) + 1),
                   100. * dev_correct / total, dev_correct, total))

    print(dev_losses)
    print(dev_accuracies)


def test(model: CNN, loss_function: nn.NLLLoss, device: torch.device,
         test_loader: torch.utils.data.DataLoader) -> None:
    """Testing loop for the CNN."""
    test_loss = 0
    test_correct = 0
    total = 0

    with torch.no_grad():
        for batch_num, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            # WRITE CODE HERE
            pred = model(data)
            loss = loss_function(pred, target)

            total += len(data)
            test_loss += loss.item()
            test_correct += (pred.argmax(1) ==
                             target).type(torch.float).sum().item()

            print('Evaluating: Batch %d/%d: Loss: %.4f | Test Acc: %.3f%% (%d/%d)' %
                  (batch_num+1, len(test_loader), test_loss / (batch_num + 1),
                   100. * test_correct / total, test_correct, total))


def main(split: bool):
    "Handles the main loop of the CNN creation process."
    if split:
        split_to_folders()
    train_loader, dev_loader, test_loader = create_loaders()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR,
                           weight_decay=WEIGHT_DECAY)
    # optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM)
    loss_function = nn.NLLLoss()

    training(model, loss_function, optimizer, device, train_loader, dev_loader)
    test(model, loss_function, device, test_loader)
    torch.save(model, MODELS_DIR + 'entire_model.pth')


if __name__ == "__main__":
    main(True)
