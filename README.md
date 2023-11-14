# handwriting-improver

## Description

Write a upper case letter in the English alphabet and see if an AI can understand it. Can be used to improve your handwriting or for playing around to see how an image recognition machine learning algorithm might work. You can save the letter you created, upload an image for the model to guess, or use the canvas to make an image and send it to the model. Uses FastAPI and uvicorn for the server side and PyTorch to make the CNN.

## Table of Contents

- [Data](#data)
- [To-do](#to-do)
- [Deployment](#deployment)
- [Layout](#layout)
- [Credits](#credits)

# Data

The data used is handwritten English upper-case letters. The dataset has been compiled by Sachin Patel and can be downloaded [here](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format).

# To-do

- [ ] Make background image work when running with uvicorn
- [ ] Improve the look of the website
- [ ] Maybe train a new, better model
- [ ] Consider removing the edges of submitted images (box around the given letter) so that we can resize/center image better

# Deployment

Deployment: [here (to-do)](https://todo/)

## Layout

The general look:

![Layout of the page (to-do)](./app/img/layout.PNG)

## Credits

Background image by Pete Linforth from [pixabay.com](https://pixabay.com/photos/connection-hand-human-robot-touch-3308188/) under the Creative Commons Zero (CC0) license.

Dataset by Sachin Patel from [kaggle.com](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format) under the Creative Commons Zero (CC0: Public Domain) license.
