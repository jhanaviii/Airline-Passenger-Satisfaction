# Airline Passenger Satisfaction Classification

Successfully developed a machine learning model to predict Airline Passenger Satisfaction by building an end-to-end MLOps pipeline. It integrates DVC for data versioning, a Dockerfile for containerization, and CI/CD using GitHub Actions for automated deployment.

## 🛠️ Tech Stack

### Machine Learning & Data Science
- **Python 3.8+**
- **scikit-learn** – Machine learning algorithms
- **pandas** – Data manipulation and analysis
- **numpy** – Numerical computing
- **matplotlib** – Data visualization
- **seaborn** – Statistical data visualization
- **plotly** – Interactive visualizations

### Advanced ML Libraries
- **XGBoost** – Gradient boosting framework
- **LightGBM** – Gradient boosting framework
- **CatBoost** – Gradient boosting on decision trees
- **imbalanced-learn** – Handling imbalanced datasets

### MLOps & Experiment Tracking
- **MLflow** – ML lifecycle management and experiment tracking
- **Optuna** – Hyperparameter optimization
- **DVC** – Data version control and ML pipelines

### Deployment & DevOps
- **Docker** – Containerization
- **Azure Web App Service** – Cloud deployment
- **GitHub Actions** – CI/CD pipeline
- **Flask / Streamlit** – Web application frameworks

### Development Environment
- **Jupyter Notebook** (99.4% of codebase)
- **Git** – Version control

---

## 📋 Prerequisites

- Python 3.8+
- Git
- Docker (optional)

---

## ⚡ Quick Setup

### 1. Clone Repository
```bash
git clone https://github.com/SayamAlt/Airline-Passenger-Satisfaction-Classification.git
cd Airline-Passenger-Satisfaction-Classification
````

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
pip install optuna mlflow dvc streamlit flask
pip install xgboost lightgbm catboost
pip install plotly imbalanced-learn
```

---

## 🏃‍♂️ Running the Project

### Option 1: Jupyter Notebook (Recommended)

```bash
jupyter notebook
```

Open the main `.ipynb` file and run cells sequentially.

### Option 2: MLflow Experiment Tracking

```bash
mlflow ui
```

Visit `http://localhost:5000` to view experiments.

### Option 3: Web Application

```bash
streamlit run app.py  # or flask run
```

---

## 📊 Data Setup

### Using DVC (Data Version Control)

```bash
dvc init
dvc pull
```

### Manual Setup

Download the airline passenger satisfaction dataset from Kaggle and place it in the `data/` folder.

---

## 🏗️ Project Structure

```
├── data/                 # Dataset files
├── notebooks/            # Jupyter notebooks (99.4% of code)
├── src/                  # Source code
├── models/               # Trained models
├── mlruns/               # MLflow experiments
├── requirements.txt      # Dependencies
├── Dockerfile            # Container configuration
├── dvc.yaml              # DVC pipeline
├── .github/workflows/    # CI/CD pipeline
└── README.md             # This file
```

---

## 🤖 Machine Learning Models

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM
* CatBoost
* Support Vector Machine
* Neural Networks

---

## 🎯 Key Features

* **Passenger Demographics:** Gender, Age, Customer Type
* **Flight Details:** Distance, Class, Type of Travel
* **Service Ratings (1–5 scale):** WiFi, Food, Seat Comfort, Entertainment, Cleanliness
* **Operational Metrics:** Departure/Arrival Delays, Gate Location, Baggage Handling

---

## 📈 Target Variable

Predicts passenger satisfaction:
**Satisfied** vs **Neutral or Dissatisfied**

---

## 🔧 MLOps Pipeline Features

* Data Versioning with **DVC**
* Experiment Tracking with **MLflow**
* Hyperparameter Optimization with **Optuna**
* Containerization with **Docker**
* CI/CD Pipeline using **GitHub Actions**
* Cloud Deployment on **Azure Web App Service**
* Model Registry and Versioning

---

## 🚨 Troubleshooting

**Module not found errors:**

```bash
pip install [missing-module-name]
```

**Data issues:**

* Ensure dataset is in the `data/` folder
* Check file paths in notebooks and scripts

**MLflow issues:**

```bash
mlflow server --host 0.0.0.0 --port 5000
```

**Docker issues:**

```bash
docker build -t airline-satisfaction .
docker run -p 8000:8000 airline-satisfaction
```

---

## 📊 Performance

* High accuracy with optimized hyperparameters
* End-to-end MLOps workflow implemented
* Fully automated and production-ready deployment pipeline

---

## 📄 License

This project is open source and available under the **MIT License**.
