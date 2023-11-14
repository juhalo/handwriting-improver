FROM python:3.11.6-bullseye

COPY . /app
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir -f https://download.pytorch.org/whl/torch/ -f https://download.pytorch.org/whl/torchaudio/ -f https://download.pytorch.org/whl/torchvision/

WORKDIR /app/app

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
