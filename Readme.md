# Student Exam Performance Indicator

A machine learning web app that predicts a student's **math score** based on demographic and academic factors. Built with Flask and deployed on Render.

**Live demo:** [your-app-name.onrender.com](https://your-app-name.onrender.com) *(replace with your Render URL)*

---

## Overview

This project takes a student's gender, ethnicity, parental education level, lunch type, test preparation status, and their reading/writing scores, and predicts their expected math score out of 100. It demonstrates an end-to-end ML workflow — data ingestion, preprocessing, model training, and deployment as a web service.

---

## Problem Statement

Educators and institutions often want to understand how socio-academic factors relate to student performance in mathematics. This project builds a regression model that estimates a student's math score from other available attributes, which can help identify students who might benefit from additional support.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.x |
| ML / Data | scikit-learn, pandas, numpy |
| Backend | Flask |
| Server | Gunicorn |
| Deployment | Render |
| Frontend | HTML, CSS |

---

## Project Structure

```
student-exam-performance/
├── app.py                     # Flask application entry point
├── requirements.txt           # Python dependencies
├── Procfile                    # Render/Gunicorn start command
├── setup.py                    # Package configuration
│
├── artifacts/
│   ├── model.pkl                # Trained regression model
│   ├── preprocessor.pkl         # Fitted preprocessing pipeline
│   └── data.csv                 # Raw/processed dataset
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py     # Loads and splits raw data
│   │   ├── data_transformation.py # Feature encoding & scaling
│   │   └── model_trainer.py       # Trains and evaluates models
│   ├── pipeline/
│   │   ├── train_pipeline.py      # End-to-end training pipeline
│   │   └── predict_pipeline.py    # Loads artifacts & makes predictions
│   ├── exception.py               # Custom exception handling
│   ├── logger.py                  # Logging configuration
│   └── utils.py                   # Helper functions (save/load objects, etc.)
│
├── templates/
│   └── home.html               # Prediction form UI
│
└── notebooks/
    └── EDA_and_Model_Training.ipynb  # Exploratory data analysis & experiments
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

1. **Data Ingestion** – Reads raw data and splits it into training and test sets.
2. **Data Transformation** – Applies one-hot encoding to categorical features and standard scaling to numerical features, saved as `preprocessor.pkl`.
3. **Model Training** – Trains multiple regression algorithms (e.g. Linear Regression, Random Forest, XGBoost, CatBoost) and selects the best performer using R² score, saved as `model.pkl`.
4. **Prediction Pipeline** – Loads the saved preprocessor and model to transform new input and generate predictions.

### Model Performance

| Model | R² Score |
|---|---|
| _Best model_ | _xx.xx %_ |

*(Fill in with your actual evaluation results)*

---

## Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/student-exam-performance.git
cd student-exam-performance
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

1. Open the web app.
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
[GitHub](https://github.com/your-username) · [LinkedIn](https://linkedin.com/in/your-profile)