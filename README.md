# ✈️ Airline Passenger Satisfaction Classification

A complete **MLOps pipeline** for predicting airline passenger satisfaction using machine learning. The project includes hyperparameter optimization, experiment tracking, and deployment-ready architecture.

---

## ⚡ Quick Setup

### ✅ Prerequisites

- Python 3.8+
- Git

---

## 🛠 Installation

### 1. Clone the repository
```bash
git clone https://github.com/SayamAlt/Airline-Passenger-Satisfaction-Classification.git
cd Airline-Passenger-Satisfaction-Classification
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
pip install optuna mlflow dvc streamlit flask
pip install xgboost lightgbm catboost
pip install plotly imbalanced-learn
```

---

## 🚀 Running the Project

### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook
```
- Open the main `.ipynb` file and run all cells sequentially.

### Option 2: Python Scripts
```bash
python main.py
```

---

## 🔬 MLflow Experiment Tracking
```bash
mlflow ui
```
- Access experiments at: [http://localhost:5000](http://localhost:5000)

---

## 🌐 Streamlit Web App (if available)
```bash
streamlit run app.py
```

---

## 📂 Data Setup

### Option 1: Using DVC
```bash
dvc init
dvc pull
```

### Option 2: Manual Download
- Download the dataset from [Kaggle](https://www.kaggle.com/datasets)  
- Place the dataset inside the `data/` directory.

---

## 📁 Project Structure

```
├── data/                 # Dataset files
├── notebooks/            # Jupyter notebooks
├── src/                  # Source code
├── models/               # Trained models
├── requirements.txt      # Dependencies
├── dvc.yaml              # DVC pipeline configuration
├── mlruns/               # MLflow experiment logs
└── README.md             # Project documentation
```

---

## 🔧 Features

- **Data Preprocessing**: Handles missing values, encodes categorical variables
- **Feature Engineering**: Selection and transformation of features
- **Model Training**: Multiple ML models with hyperparameter tuning using Optuna
- **Experiment Tracking**: MLflow integration for tracking experiments
- **Model Evaluation**: Detailed performance metrics and visualizations
- **Deployment**: Docker-based containerization and Azure Web App deployment

---

## 🧠 Models Used

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Support Vector Machine (SVM)

---

## 🎯 Target Variable

Predicts whether a passenger is:

- `Satisfied`
- `Neutral or Dissatisfied`

---

## 🔑 Key Features

- Demographics: `Gender`, `Age`, `Customer Type`
- Travel Info: `Flight Distance`, `Class`, `Type of Travel`
- Services Ratings (1–5): `WiFi`, `Food`, `Seat Comfort`, `Entertainment`, etc.
- Delay Details: `Departure/Arrival Delay`

---

## 📊 Results

- High accuracy achieved through extensive hyperparameter tuning
- End-to-end MLOps pipeline
- Deployed successfully using Azure Web App

---

## 🧯 Troubleshooting

### Module Not Found Errors:
```bash
pip install [missing-module-name]
```

### Data Issues:
- Ensure dataset exists in the `data/` folder
- Check file paths in notebooks and scripts

### MLflow Issues:
```bash
mlflow server --host 0.0.0.0 --port 5000
```

---

## 🌍 Live Demo

Deployed App:  
[https://airline-passenger-satisfaction-akevhzeuh8btgffr.canadacentral-01.azurewebsites.net](https://airline-passenger-satisfaction-akevhzeuh8btgffr.canadacentral-01.azurewebsites.net)

---

## 📄 License

This project is licensed under the **MIT License**.
