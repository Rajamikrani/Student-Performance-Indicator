# Student Exam Performance Indicator

A machine learning web app that predicts a student's **math score** based on demographic and academic factors. Built with Flask and deployed on Render.

**Live demo:** [student-performance-indicator-w8hn.onrender.com/predictdata](https://student-performance-indicator-w8hn.onrender.com/predictdata)

---

## Overview

This project takes a student's gender, ethnicity, parental education level, lunch type, test preparation status, and their reading/writing scores, and predicts their expected math score out of 100. It demonstrates an end-to-end ML workflow — data ingestion, preprocessing, model training, model selection, and deployment as a web service.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| ML / Data | scikit-learn, pandas, numpy, catboost, xgboost |
| Backend | Flask |
| Server | Gunicorn |
| Deployment | Render |
| Frontend | HTML, CSS |

---

## Project Structure

```
Student-Performance-Indicator/
├── app.py                          # Flask application entry point
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version for Render
├── render.yaml                     # Render service configuration
├── setup.py                        # Package configuration
│
├── artifact/
│   ├── model.pkl                     # Trained regression model
│   ├── preprocesser.pkl              # Fitted preprocessing pipeline
│   ├── data.csv                      # Cleaned dataset
│   ├── train.csv                     # Training split
│   └── test.csv                      # Test split
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py          # Loads, splits, and saves data
│   │   ├── data_transformation.py     # Feature encoding & scaling pipeline
│   │   └── model_trainer.py           # Trains, evaluates, and selects best model
│   ├── pipelines/
│   │   ├── train_pipeline.py          # End-to-end training pipeline
│   │   └── predict_pipeline.py        # Loads artifacts & makes predictions
│   ├── exception.py                   # Custom exception handling
│   ├── logger.py                      # Logging configuration
│   └── utils.py                       # Helper functions (save/load objects, evaluation)
│
├── templates/
│   ├── index.html                  # Landing page
│   └── home.html                   # Prediction form UI
│
└── notebook/
    ├── EDA STUDENT PERFORMANCE.IPYNB   # Exploratory data analysis
    ├── MODEL_TRAINING.ipynb            # Model training & comparison
    └── data/stud.csv                   # Raw dataset
```

---

## Dataset

The model is trained on the **Students Performance in Exams** dataset, containing the following features:

| Feature | Description |
|---|---|
| `gender` | Male / Female |
| `race_ethnicity` | Group A–E |
| `parental_level_of_education` | Highest education level of parent(s) |
| `lunch` | Standard / Free or reduced |
| `test_preparation_course` | Completed / None |
| `reading_score` | Score out of 100 |
| `writing_score` | Score out of 100 |
| `math_score` | **Target variable** — score out of 100 |

---

## Machine Learning Pipeline

1. **Data Ingestion** – Reads the raw dataset and splits it into training and test sets (`train.csv`, `test.csv`).
2. **Data Transformation** – Applies median imputation + standard scaling to numerical features and one-hot encoding + scaling to categorical features, saved as `preprocesser.pkl`.
3. **Model Training & Selection** – Trains multiple regression algorithms and evaluates each on R² score against the test set. The best-performing model is saved as `model.pkl`.
4. **Prediction Pipeline** – Loads the saved preprocessor and model to transform new form input and generate a math score prediction.

### Model Comparison

| Model | R² Score |
|---|---|
| **Ridge** ✅ (selected) | **0.8759** |
| Linear Regression | 0.8725 |
| CatBoosting Regressor | 0.8521 |
| Random Forest | 0.8473 |
| AdaBoost | 0.8281 |
| XGBoost | 0.8118 |
| Lasso | 0.8103 |
| KNN | 0.7723 |
| Support Vector Machine | 0.7268 |
| Decision Tree | 0.7222 |

Ridge Regression gave the best generalization performance and was selected as the production model.

---

## Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/Rajamikrani/Student-Performance-Indicator.git
cd Student-Performance-Indicator
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Flask app
```bash
python app.py
```

The app will be available at `http://127.0.0.1:5000`.

---

## Deployment (Render)

This app is deployed on [Render](https://render.com) as a web service.

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn app:app`

Render automatically redeploys on every push to the `main` branch.

> **Note:** The free tier spins down after periods of inactivity, so the first request after idle time may take 30–60 seconds to respond.

---

## Usage

1. Visit the [prediction form](https://student-performance-indicator-w8hn.onrender.com/predictdata).
2. Fill in the student's details: gender, ethnicity, parental education, lunch type, test preparation status, and reading/writing scores.
3. Click **Predict math score** to get the estimated math score.

---

## Future Improvements

- Add model explainability (e.g. SHAP) to show which factors influenced a prediction
- Add input validation and error handling on the form
- Containerize with Docker for consistent deployments
- Add unit tests for the pipeline components
- Track experiments with MLflow

---

## Author

**Raja**
BCA Student, Tribhuvan University, Kathmandu, Nepal
[GitHub](https://github.com/Rajamikrani)