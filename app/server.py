"""Provides server functionality for the app."""
from string import ascii_uppercase
import numpy as np
from fastapi import FastAPI, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from PIL import Image
from PIL import ImageOps

import torch
from torchvision import transforms

from cnn_model import CNN


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory='templates')


@app.get("/")
def home(request: Request):
    """Returns the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


CUR_DICT_PATH = "./models/only_dict.pth"

model = CNN()
model.load_state_dict(torch.load(CUR_DICT_PATH))


def get_prediction(img):
    """Does the actual prediction for an image."""
    with torch.no_grad():
        model.eval()
        output = model(img)
        index = output.data.cpu().numpy().argmax()
        classes = list(ascii_uppercase)
        pred = classes[index]
        return_dict = {'prediction': pred}
        return return_dict


@app.post("/predict/")
def predict(file: UploadFile):
    """When making POST request to /predict, runs the CNN for the provided image."""
    img = Image.open(file.file)
    img_numpy = np.asarray(img)
    num_black = (img_numpy == 0).sum()
    num_white = (img_numpy == 255).sum()
    is_not_inverted = num_black < num_white

    if is_not_inverted:
        # https://stackoverflow.com/questions/2498875/how-to-invert-colors-of-image-with-pil-python-imaging
        if img.mode == 'RGBA':
            r, g, b, _ = img.split()
            rgb_image = Image.merge('RGB', (r, g, b))
            img_invert = ImageOps.invert(rgb_image)
        else:
            img_invert = ImageOps.invert(img)
    else:
        img_invert = img
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(28),
        transforms.ToTensor()
    ])

    img_transform = transform(img_invert).float()
    img_transform = img_transform.unsqueeze_(0)

    prediction = get_prediction(img_transform)

    return prediction


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
