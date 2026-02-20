# EEG Seizure Risk Prediction System using Machine Learning and FastAPI

## Overview

This project is a Medical AI system that predicts seizure risk using EEG brain signal features.  
It uses supervised learning, pipeline architecture, hyperparameter tuning, and FastAPI deployment.

The system accepts EEG features and returns seizure risk prediction with confidence score.

---

## Dataset

Dataset: BEED (Bangalore EEG Epilepsy Dataset)  
Instances: 8000  
Features: 16 EEG features  
Target: Seizure (1) or Non-Seizure (0)  
Missing values: None  

---

## Technologies Used

- Python
- Scikit-learn
- FastAPI
- Pandas
- NumPy
- Joblib

---

## Machine Learning Workflow

- Data Loading
- Train-Test Split
- Pipeline Creation
- Feature Scaling using StandardScaler
- Random Forest Classifier
- Hyperparameter Tuning using GridSearchCV
- Model Saving using Joblib
- API Deployment using FastAPI

---

## Project Structure

```
EEG_Seizure_Project
│
├── dataset.csv
├── train.py
├── main.py
├── model.pkl
├── requirements.txt
├── README.md
└── venv
```

---

## How to Run

### Step 1: Install dependencies

```
pip install -r requirements.txt
```

### Step 2: Train model

```
python train.py
```

### Step 3: Run FastAPI server

```
uvicorn main:app --reload
```

### Step 4: Open browser

```
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### Health Check

```
GET /
```

### Model Info

```
GET /model-info
```

### Single Prediction

```
POST /predict
```

### Batch Prediction

```
POST /predict-batch
```

---

## Example Output

```
{
  "prediction": 1,
  "result": "Seizure Risk Detected",
  "confidence_score": 0.94
}
```

---

## Key Features

- Supervised Learning Model
- Pipeline Architecture
- Hyperparameter Optimization
- Production-ready FastAPI Deployment
- Confidence Score Prediction
- Batch Prediction Support

---

## Author

Vaishnavi Reddy  
Machine Learning Student