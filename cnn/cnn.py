"""Trains the model used for the prediction"""
# import torch
# import torch.optim as optim
# import torch.utils.data
# import torch.backends.cudnn as cudnn
# import torchvision
# from torchvision import transforms, datasets
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import math
# # import pandas as pd
import random as ra
import shutil
import os
import string
from my_constants import IMAGES_DIR, TRAIN_DIR, TEST_DIR, DEV_DIR


def split_to_folders():
    "Splits the images to train, dev, test folders"
    alphabets = list(string.ascii_uppercase)
    alphabet_paths = [IMAGES_DIR + alphabet + "/" for alphabet in alphabets]
    paths = [TRAIN_DIR, TEST_DIR, DEV_DIR]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
    for alphabet_path in alphabet_paths:
        image_file_names = os.listdir(alphabet_path)
        for image_file_name in image_file_names:
            image_full_path = alphabet_path + image_file_name
            if os.path.isfile(image_full_path):
                division = ra.randint(1, 4)
                if division == 1 or division == 2:
                    shutil.copyfile(image_full_path,
                                    TRAIN_DIR + image_file_name)
                if division == 3:
                    shutil.copyfile(image_full_path, DEV_DIR + image_file_name)
                if division == 4:
                    shutil.copyfile(image_full_path,
                                    TEST_DIR + image_file_name)


def main(split: bool):
    "Handles the main loop of the CNN creation process."
    if split:
        split_to_folders()


if __name__ == "__main__":
    main(True)
