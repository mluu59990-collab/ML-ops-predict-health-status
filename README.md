# MLOps for Health Status Prediction

## Overview
This project builds an end-to-end MLOps pipeline for predicting health status using machine learning.  
It covers the full workflow from data preprocessing and model training to experiment tracking, version control, API deployment, and automation.

The goal of this project is not only to build a prediction model, but also to organize the machine learning lifecycle in a reproducible and scalable way.

---

## Objectives
- Predict health status from input health-related features
- Build a reproducible machine learning workflow
- Track experiments and model performance
- Version datasets and pipelines
- Containerize the application for deployment
- Serve predictions through an API
- Automate workflows using CI/CD

---

## Project Architecture
The project includes the following components:

- **Data preprocessing**: cleaning, transforming, and preparing data
- **Model training**: training machine learning models for health status prediction
- **Model evaluation**: measuring performance using suitable metrics
- **Experiment tracking**: logging runs, parameters, and results with MLflow
- **Data & pipeline versioning**: managing data and pipeline changes with DVC
- **API deployment**: serving the trained model with FastAPI
- **Containerization**: packaging the app using Docker
- **Automation**: running workflows with GitHub Actions

---

## Technologies Used
- **Python**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **MLflow**
- **DVC**
- **FastAPI**
- **Docker**
- **GitHub Actions**

---

## Project Structure
```bash
.
├── data/                 # raw and processed data
├── notebooks/            # exploratory notebooks
├── src/                  # source code
│   ├── preprocessing/    # data preprocessing scripts
│   ├── training/         # model training scripts
│   ├── evaluation/       # evaluation scripts
│   └── api/              # FastAPI application
├── models/               # saved models
├── dvc.yaml              # DVC pipeline definition
├── requirements.txt      # dependencies
├── Dockerfile            # Docker configuration
├── .github/workflows/    # GitHub Actions workflows
└── README.md
