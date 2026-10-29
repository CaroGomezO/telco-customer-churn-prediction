FROM python:3.11-slim-bookworm

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline.py train.py inference.py predict.py ./

