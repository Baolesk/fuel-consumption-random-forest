# Fuel Consumption Prediction using Random Forest

This project predicts vehicle fuel consumption (L/100km) using telemetry data collected from OBD-II.

## Features
- Engine RPM
- Horsepower at the wheels

## Target
Litres Per 100 Kilometer (L/100km)

## Model
Random Forest Regressor

## Dataset
Data collected from real driving conditions.

## Run

Install dependencies

pip install -r requirements.txt

Train model

python src/train.py --data data/sample.csv

Predict

python src/predict.py --rpm 2200 --hp 15

## Research Publication

This project was presented at the Young Scientific Conference (YSC).

Paper: docs/paper.pdf  
Poster: docs/poster.png  
Certificate: docs/certificate.pdf