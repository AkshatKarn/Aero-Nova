# Aero-Nova: AI-Driven Air Quality & Mobility Nowcasting System

## Overview

Aero-Nova is an AI-powered environmental intelligence platform designed to forecast Air Quality Index (AQI) using real-time and historical spatio-temporal data. The system combines machine learning, data analytics, visualization, and predictive modeling to help users, policymakers, and organizations understand pollution trends and make data-driven decisions.

The platform integrates weather conditions, traffic density, industrial emissions, and mobility data to generate accurate AQI predictions and simulate the impact of environmental policies such as traffic reduction and electric vehicle adoption.

---

# Key Features

## AI-Powered AQI Forecasting

* Predicts future AQI levels using machine learning models.
* Uses environmental and mobility datasets for spatio-temporal forecasting.
* Supports short-term nowcasting and trend analysis.

## Predictive Analytics

* Implements Gradient Boosting and XGBoost models for AQI prediction.
* Supports future integration of Temporal Fusion Transformer (TFT) models.
* Detects pollution spikes and high-risk environmental conditions.

## Interactive Dashboards

* Real-time dashboards built using Streamlit and Power BI.
* Dynamic visualization of AQI trends, pollution hotspots, and mobility patterns.
* User-friendly interface for analysis and monitoring.

## Policy Simulation Engine

* Simulates environmental policy scenarios such as:

  * Traffic reduction
  * EV adoption
  * Industrial regulation
  * Mobility restrictions
* Helps evaluate the potential impact on air quality.

## Health Advisory Chatbot

* NLP-powered chatbot for personalized air quality recommendations.
* Provides health guidance based on AQI levels.
* Suggests preventive measures for sensitive groups.

## Data Integration Pipeline

* Integrates:

  * Weather datasets
  * Traffic datasets
  * AQI datasets
  * Industrial emission data
* Automated preprocessing and feature engineering pipeline.

---

# Problem Statement

Urban air pollution has become a major environmental and public health challenge. Traditional AQI monitoring systems mainly provide static or delayed information, making proactive decision-making difficult.

Aero-Nova addresses this challenge by:

* Predicting future AQI conditions
* Analyzing environmental and mobility factors
* Providing intelligent visual insights
* Supporting policy-level decision making

---

# Objectives

* Develop an AI-powered AQI nowcasting platform.
* Forecast pollution levels using machine learning.
* Build real-time analytical dashboards.
* Simulate environmental policy impacts.
* Provide actionable public health insights.
* Improve accessibility of environmental intelligence.

---

# Tech Stack

## Programming Languages

* Python
* SQL

## Machine Learning & Data Science

* XGBoost
* Scikit-learn
* Pandas
* NumPy

## Visualization & Dashboarding

* Power BI
* Streamlit
* Plotly
* Matplotlib

## NLP & Chatbot

* Rasa
* NLTK

## Database & Storage

* MySQL
* CSV Datasets

## Development Tools

* VS Code
* Google Colab
* Git & GitHub

---

# System Architecture

The Aero-Nova platform follows a modular AI analytics architecture:

1. Data Collection Layer

   * Weather APIs
   * AQI datasets
   * Traffic datasets
   * Industrial data

2. Data Processing Layer

   * Data cleaning
   * Missing value handling
   * Feature engineering
   * Data normalization

3. Machine Learning Layer

   * AQI prediction models
   * Trend analysis
   * Scenario simulation

4. Analytics & Visualization Layer

   * Streamlit dashboard
   * Power BI reports
   * Interactive charts

5. User Interaction Layer

   * Health advisory chatbot
   * AQI monitoring interface
   * Policy simulation tools

---

# Machine Learning Workflow

## Data Preprocessing

* Removed missing and duplicate records.
* Standardized environmental parameters.
* Performed feature scaling and transformation.

## Feature Engineering

Features considered:

* Temperature
* Humidity
* Wind speed
* Traffic density
* Industrial activity
* Vehicle emissions
* Historical AQI values

## Model Training

Models used:

* Gradient Boosting
* XGBoost Regressor

## Evaluation Metrics

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

---

# Dashboard Features

## AQI Monitoring Dashboard

* Current AQI trends
* Pollution category visualization
* City-wise AQI analysis

## Predictive Dashboard

* AQI forecasting charts
* Trend prediction visualization
* Risk-level indicators

## Mobility Analytics Dashboard

* Traffic vs AQI analysis
* Emission trend comparison
* Environmental impact tracking

## Policy Simulation Dashboard

* Simulated AQI reduction analysis
* Comparative environmental scenarios
* Predictive policy insights

---

# Folder Structure

```bash
Aero-Nova/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── models/
│   ├── trained_models/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── model_training.ipynb
│
├── dashboard/
│   ├── app.py
│   ├── powerbi_reports/
│
├── chatbot/
│   ├── rasa_bot/
│
├── utils/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── prediction.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/your-username/Aero-Nova.git
cd Aero-Nova
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Run Streamlit Dashboard

```bash
streamlit run app.py
```

## Run Model Training

```bash
python train_model.py
```

## Run Chatbot

```bash
rasa run
```

---

# Future Enhancements

* Integration with live AQI APIs
* Deep learning-based forecasting models
* Mobile application support
* Multi-city pollution prediction
* IoT sensor integration
* Real-time alert notification system
* Cloud deployment using AWS or Azure

---

# Use Cases

* Smart city analytics
* Environmental monitoring
* Government policy planning
* Public health advisory systems
* Traffic management optimization
* Sustainability research

---

# Project Impact

Aero-Nova aims to bridge the gap between environmental monitoring and intelligent decision-making by combining AI, predictive analytics, and interactive visualization. The platform enables proactive pollution management and promotes healthier urban living through data-driven environmental intelligence.

---

# Author

## Rishika Jain

AI & Data Analytics Enthusiast

* Python Developer
* Machine Learning Enthusiast
* Data Visualization & BI Practitioner

LinkedIn: linkedin.com/in/rishika-jain-692422316

---

# License

This project is developed for aca
