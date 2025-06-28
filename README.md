---

# 🏎️ Formula 1 Race Outcome Prediction

Predicting whether a Formula 1 driver will finish on the podium (top 3) using historical data, advanced feature engineering, and machine learning.

---

## 📌 Table of Contents

* [Introduction](#introduction)
* [Key Terminology](#key-terminology)
* [Problem Statement](#problem-statement)
* [Data Sources](#data-sources)
* [Methodology](#methodology)
* [Exploratory Data Analysis](#exploratory-data-analysis)
* [Statistical Testing](#statistical-testing)
* [Feature Engineering](#feature-engineering)
* [Modeling](#modeling)
* [Deployment](#deployment)
* [How to Run](#how-to-run)
* [Conclusion](#conclusion)

---

## Introduction

The world of Formula 1 is rich with data—qualifying rounds, race conditions, team strategies, driver performances, and track histories. This project leverages that data to predict one thing: will a driver finish on the podium?

---

## 🏁 Key Terminology

- **Driver** — The individual competing in the race. Each F1 team has two drivers.
- **Constructor (Team)** — The organization that builds and races the car. Examples: Mercedes, Ferrari, Red Bull Racing.
- **Grand Prix (Race, Round)** — A single event in the F1 calendar, typically held over a weekend, consisting of practice sessions, qualifying, and the main race.
- **Qualifying** — A session that determines the starting grid for the race. A better qualifying position often improves race performance.
- **Grid Position**: The starting position of a driver in the race.
- **Podium** — The top 3 finishers in a race — 1st, 2nd, and 3rd place. These are the drivers who physically stand on the podium after the race and receive trophies.
- **Pole Position** — The first position on the starting grid, awarded to the fastest qualifier.
- **Pit Stop** — When a driver enters the pit lane to change tyres or fix minor issues. Time-consuming, but sometimes strategically vital. The pit stop itself ideally takes 2-3 seconds, but the whole process of entering and exiting the pits lasts about 20-25 seconds.
- **DNF (Did Not Finish)** — When a driver does not complete the race due to a crash, mechanical failure, or other issue.

---

##  Problem Statement

**Goal**: Predict whether a driver will finish on the podium using a historical dataset.

This is a **binary classification problem** with significant class imbalance (only \~15% of drivers finish in the top 3).

---

##  Data Sources

* Main datasets: https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020
* Pit Stops: https://www.kaggle.com/datasets/akashrane2609/formula-1-pit-stop-dataset
* Weather API and race metadata
* Historical driver/team performance

---

##  Methodology

1. Data cleaning & merging
2. Feature engineering (both static and temporal)
3. Statistical testing
4. Model training with hyperparameter tuning
5. Evaluation with real-world test set (2024 races)
6. Deployment-ready pipeline + API

---

##  Exploratory Data Analysis

Exploration included:

* Class imbalance check
* Grid position impact
* Team and driver podium rates
* Circuit-based performance
* Weather condition summaries
* Global distribution of F1 circuits

---

##  Statistical Testing

We used:

* **Chi-Squared Tests**: For independence of categorical features.
* **Mann-Whitney U Tests**: For differences in feature distributions across classes.
* Results informed feature selection.

---

##  Feature Engineering

Key features:

* **Driver Experience** (race count)
* **Recent Performance** (last 3 races)
* **Rolling Average Finish**
* **Constructor Podium Rate**
* **Track-Specific Averages**
* **Weather Flags** (wet, windy, hot, cold)
* **Binary-Encoded Categorical Features**

All feature engineering is encapsulated in a reusable `F1DataPreprocessor` transformer.

---

##  Modeling

### 1. **Baseline Models**

* Logistic Regression
* Random Forest

### 2. **Class Imbalance Handling**

* Cost-sensitive learning
* SMOTE and over-sampling

### 3. **Advanced Models**

* HistGradientBoostingClassifier
* LightGBM, XGBoost, CatBoost
* Ensembles with Voting and Stacking

### 4. **Hyperparameter Tuning**

* Optuna with AUC & F1 scores

---

## 🏁 Final Model

**Best Model**: Random Forest with Optuna-tuned hyperparameters
**Test AUC**: *937*
**Test F1**: *0.72*
**Test Precision**: *0.64*

---

##  Deployment

### Components

* **Custom Transformer** (`F1DataPreprocessor`)
* **Custom Pipeline** (`F1Pipeline`)
* **Joblib Model Saving**:

  ```python
  joblib.dump(preprocessor, "f1_preprocessor.pkl")
  joblib.dump(model, "models/model.pkl")
  ```

### FastAPI Inference Server

A FastAPI service is provided for predicting podium chances in real-time.

### Dockerized API

The API is dockerized for easy deployment:

```bash
docker build -t f1-predictor .
docker run -p 8000:8000 f1-predictor
```

---

## 🛠️ How to Run

1. Clone the repo:

   ```bash
   git clone https://github.com/wsiqz/formula-1.git
   cd formula-1
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the pipeline notebook:

   ```
   notebooks/f1.ipynb
   ```

4. Train and export the model:

   ```python
   joblib.dump(preprocessor, "f1_preprocessor.pkl")
   joblib.dump(model, "models/model.pkl")
   ```

5. Start FastAPI server:

   ```bash
   uvicorn app.main:app --reload
   ```

---

## 🧾 Conclusion

This project demonstrates how domain knowledge, careful feature engineering, and rigorous modeling techniques can be combined to solve real-world predictive problems—even in complex, dynamic environments like Formula 1.

---
