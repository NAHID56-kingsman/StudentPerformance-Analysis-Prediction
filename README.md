# 🎓 Student Performance Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)


*A machine learning system that predicts student exam scores and pass/fail outcomes based on study patterns and behaviors*

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Models](#-models) • [Results](#-results)

</div>

---

## 📖 Overview

This project implements a dual-prediction machine learning system designed to forecast student academic performance. Using historical data on study habits, attendance, and past performance, the system predicts:

1. **Final Exam Score** (Regression Task)
2. **Pass/Fail Outcome** (Classification Task)

The system is designed to help educators identify at-risk students early and provide targeted interventions.

---

## ✨ Features

- 🎯 **Dual Prediction Models**: Regression for scores, classification for pass/fail
- 📊 **Data Preprocessing Pipeline**: Automatic encoding and scaling
- 🔍 **Real-time Predictions**: Interactive input system for instant results
- 📈 **Comprehensive Metrics**: Multiple evaluation metrics for model performance
- 🧹 **Clean Architecture**: Modular and well-documented code
- 💾 **Scalable Design**: Easy to extend with additional features

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **Data Processing** | pandas, numpy |
| **Visualization** | matplotlib |
| **Machine Learning** | scikit-learn |
| **Models** | Random Forest, Logistic Regression |

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/student-performance-prediction.git
cd student-performance-prediction
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Libraries

```txt
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

---

## 🚀 Usage

### Basic Usage

1. **Prepare your dataset**
   - Ensure your CSV file contains the required columns 

2. **Run the main script**
```bash
python student_prediction.py
```

3. **Enter student information when prompted**
```
Enter Study Hours per Week: 15
Enter Attendance Rate: 85
Enter Past Exam Score: 75
Internet Access at Home? 1=Yes// 0=No: 1
Extracurricular Activities? 1=Yes // 0=No: 1
```

4. **View predictions**
```
Predicted Final Score: 82.45
Predicted Pass/Fail: Pass
```

### Data Format

Your dataset should include the following columns:

| Column Name | Type | Description |
|------------|------|-------------|
| `Study_Hours_per_Week` | Float | Weekly study hours |
| `Attendance_Rate` | Float | Attendance percentage (0-100) |
| `Past_Exam_Scores` | Float | Previous exam scores |
| `Internet_Access_at_Home` | String | "Yes" or "No" |
| `Extracurricular_Activities` | String | "Yes" or "No" |
| `Final_Exam_Score` | Float | Target variable (score) |
| `Pass_Fail` | String | "Pass" or "Fail" |

---

## 🧠 Models

### 1. Regression Model (Score Prediction)

**Algorithm**: Random Forest Regressor with MultiOutputRegressor

**Features**:
- Handles non-linear relationships
- Robust to outliers
- Feature importance analysis

```python
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

model = MultiOutputRegressor(RandomForestRegressor(
    n_estimators=100,
    random_state=42
))
```

### 2. Classification Model (Pass/Fail Prediction)

**Algorithm**: Logistic Regression

**Features**:
- Probabilistic predictions
- Interpretable coefficients
- Fast training and prediction

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

---

## 📊 Data Preprocessing

### 1. Categorical Encoding
```python
from sklearn.preprocessing import OrdinalEncoder

oenc = OrdinalEncoder()
df[['Internet_Access_at_Home', 'Extracurricular_Activities', 'Pass_Fail']] = \
    oenc.fit_transform(df[['Internet_Access_at_Home', 'Extracurricular_Activities', 'Pass_Fail']])
```

### 2. Feature Scaling
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Study_Hours_per_Week', 'Past_Exam_Scores', 'Attendance_Rate']] = \
    scaler.fit_transform(df[['Study_Hours_per_Week', 'Past_Exam_Scores', 'Attendance_Rate']])
```

---

## 📈 Results

### Regression Metrics

| Metric | Value |
|--------|-------|
| Mean Absolute Error (MAE) | X.XX |
| Mean Squared Error (MSE) | X.XX |
| R² Score | 0.XX |

### Classification Metrics

```
              precision    recall  f1-score   support

        Fail       0.XX      0.XX      0.XX       XXX
        Pass       0.XX      0.XX      0.XX       XXX

    accuracy                           0.XX       XXX
```

**Confusion Matrix**:
```
[[TN  FP]
 [FN  TP]]
```

---

## 📁 Project Structure

```
student-performance-prediction/
│
├── student_prediction.py       # Main script
├── data/
│   └── student_data.csv        # Dataset
├── models/
│   ├── regression_model.pkl    # Saved regression model
│   └── classification_model.pkl # Saved classification model
├── notebooks/
│   └── exploration.ipynb       # Data exploration
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── LICENSE                     # MIT License
```

---

## 🔮 Future Improvements

- [ ] 📊 Advanced feature engineering and selection
- [ ] 🎨 Interactive web dashboard using Streamlit/Flask
- [ ] 💾 Model persistence with joblib
- [ ] 🔄 Cross-validation and hyperparameter tuning
- [ ] 📉 Learning curves and performance visualization
- [ ] 🌐 REST API for model deployment
- [ ] 📱 Mobile app integration
- [ ] 🤖 Deep learning models (Neural Networks)
- [ ] 📧 Email alerts for at-risk students
- [ ] 🔐 User authentication system

---


Made with ❤️ and Python

</div>
