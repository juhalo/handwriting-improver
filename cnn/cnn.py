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


def main(split: bool):
    "Handles the main loop of the CNN creation process."
    if split:
        split_to_folders()
    train_loader, dev_loader, test_loader = create_loaders()


if __name__ == "__main__":
    main(True)
