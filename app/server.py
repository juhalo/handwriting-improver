"""Provides server functionality for the app."""
from fastapi import FastAPI, UploadFile, Request
from fastapi.templating import Jinja2Templates
import uvicorn
from PIL import Image
from string import ascii_uppercase

import torch
from torchvision import transforms

from cnn.cnn import CNN


app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/")
def home(request: Request):
    """Returns the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


CUR_MODEL_PATH = "./models/entire_model.pth"

model = torch.load(CUR_MODEL_PATH)


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


@app.post("/predict")
def predict(file: UploadFile):
    """When making POST request to /predict, runs the CNN for the provided image."""
    # Read Image
    img = Image.open(file.file)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(28),
        transforms.ToTensor()
    ])
    img_transform = transform(img).float()
    img_transform = img_transform.unsqueeze_(0)

    # Use the model to generate a prediction
    prediction = get_prediction(img_transform)

    # Return the prediction as a JSON response
    return prediction


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
