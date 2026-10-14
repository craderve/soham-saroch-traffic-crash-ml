# Soham Saroch · TrafficLens

A personalized, responsive US traffic crash exploration dashboard, adapted from
[v0lencia/US-Traffic-Crash-ML-main](https://github.com/v0lencia/US-Traffic-Crash-ML-main).
The original trained XGBoost models, notebook, and MIT attribution are preserved.
The new interface calls a Flask API and runs on Vercel without a Streamlit server.

## Run the dashboard locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`. On macOS, XGBoost needs an OpenMP runtime. If libomp
is not installed, the runtime already bundled with scikit-learn can be used:

```bash
export DYLD_LIBRARY_PATH="$(python -c 'import sklearn,pathlib; print(pathlib.Path(sklearn.__file__).parent / ".dylibs")')"
python app.py
```

## Validate and deploy

```bash
python -m unittest discover -s tests -v
node --check public/app.js
vercel --prod --yes
```

Vercel detects the root Flask entrypoint `app.py`. `.python-version` pins its
runtime to Python 3.12. Production requirements use CPU-only XGBoost on Linux to
avoid unnecessary CUDA dependencies. `public/` contains the CDN-served interface.
Models load lazily and remain cached across warm requests. The first prediction
may take longer while the function starts.

API endpoints: `GET /api/health`, `GET /api/options`, and `POST /api/predict`.
The prediction endpoint validates numeric ranges, categories, and boolean inputs.
Tests cover real inference, feature mapping, invalid inputs, and oversized bodies.

Notebook dependencies are preserved in `requirements-dev.txt`. To run the
original Streamlit interface, use `streamlit_deployment/requirements.txt` and
`streamlit run streamlit_deployment/app.py` from the repository root.

## Important limitations and attribution

This is an educational adaptation, not a real-time accident forecast or navigation
safety tool. Severity means the dataset's traffic-impact category, not injury
severity or the probability that a crash will happen. Model probabilities are not
calibrated safety guarantees. Coordinates are model estimates, not observed incidents.
The original pipeline estimates wind chill as temperature minus 5 °F, fixes twilight
to Day, and offers counties independently of state. Those assumptions are retained
for compatibility and can produce geographically inconsistent scenarios.

The displayed evaluation metrics come from the original model metadata, not a new
evaluation. Original implementation and models: **Anıl Metin**, copyright 2025.
Personalized dashboard adaptation: **Soham Saroch**. See [LICENSE](LICENSE).

---

# 🚗 US Traffic Crash Machine Learning

![Python](https://img.shields.io/badge/Python-3.13.5-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.47.0-red)
![ML](https://img.shields.io/badge/ML-XGBoost-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20App-orange?logo=streamlit&logoColor=white)](https://us-traffic-crash-ml.streamlit.app/)

## 📊 Project Overview

This project is a comprehensive machine learning project developed to analyze and predict traffic accidents in the United States. The project works with over 7.7 million accident data collected between 2016-2023 and provides an interactive web application for real-time predictions.

### Main Objectives of the Project:
- **Data Analysis**: Comprehensive exploratory data analysis of US traffic accident data
- **Machine Learning**: XGBoost models for accident severity and location prediction
- **Web Application**: Streamlit application for real-time predictions
- **Explainable AI**: Model explainability with SHAP and LIME
- **Safety Insights**: Actionable insights to improve traffic safety

## 📁 Project Structure

```
├── 📄 US_Traffic_Crash_ML.ipynb                            # Main analysis notebook
├── 📊 US_Accidents_March23.csv                             # Dataset (3.06GB, not included GitHub due to large size)
├── 📋 requirements.txt                                     # Development environment dependencies
├── 🌐 streamlit_deployment/                                # Production environment
│   ├── app.py                                              # Streamlit web application
│   ├── requirements.txt                                    # Production dependencies
│   ├── severity_classifier.pkl                             # Severity prediction model
│   ├── location_regressor.pkl                              # Location prediction model
│   ├── preprocessor.pkl                                    # Data preprocessing pipeline
│   ├── model_metadata.json                                 # Model performance metrics
│   ├── feature_names.pkl                                   # Feature names
│   └── unique_counties.txt                                 # US state/county list
├── 📈 xai_plots/                                           # XAI visualizations
│   ├── classification_feature_importance.png               # Classification feature importance
│   ├── classification_permutation_importance.png           # Permutation importance
│   ├── classification_day_night_comparison.png             # Day/night comparison
│   ├── classification_weekend_weekday_comparison.png       # Weekend/weekday comparison
│   ├── regression_permutation_importance.png               # Regression permutation importance
│   ├── regression_day_night_comparison.png                 # Regression day/night
│   └── regression_weekend_weekday_comparison.png           # Regression weekend/weekday
└── .venv/                                                  # Virtual environment

```

## 🧠 Machine Learning Models & Performance

### 🎯 Model Selection and Comparison

Multiple algorithms were comprehensively tested and evaluated:

#### 📊 Classification Model Comparison Results
```python
# Cross-Validation Results (F1-Macro Score):
Model Performance Ranking:
1. Random Forest: F1-Macro = 0.9522 (117.6s)
2. HistGradientBoosting: F1-Macro = 0.8869 (613.1s)
3. XGBoost: F1-Macro = 0.8760 (400.4s)
4. KNN: F1-Macro = 0.8196 (453.5s)
5. Logistic Regression: F1-Macro = 0.6834 (1739.8s)
6. Naive Bayes: F1-Macro = 0.4578 (29.4s)

Best Performing: Random Forest (95.22% F1-Macro)
Selected for Production: XGBoost (87.60% F1-Macro)
```

#### 📊 Regression Model Comparison Results
```python
# Cross-Validation Results (R² Score & MAE):
Model Performance Ranking:
1. Random Forest Regressor: R² = 0.9991, MAE = 0.1062 (780.6s)
2. Linear Regression: R² = 0.9977, MAE = 0.1749 (80.8s)
3. Ridge: R² = 0.9977, MAE = 0.1878 (46.3s)
4. XGBoost Regressor: R² = 0.9957, MAE = 0.3121 (67.0s)
5. HistGradientBoosting Regressor: R² = 0.9950, MAE = 0.3056 (134.4s)
6. KNN Regressor: R² = 0.9153, MAE = 1.3124 (132.0s)

Best Performing: Random Forest Regressor (99.91% R²)
Selected for Production: XGBoost Regressor (99.57% R²)
```

#### 🤔 Why XGBoost was Selected Over Random Forest:
1. **Production Efficiency**: Faster inference time (67.0s vs 780.6s training)
2. **Model Interpretability**: Better SHAP integration and explainability
3. **Generalization**: Lower overfitting risk with regularization
4. **Deployment Size**: Smaller model size for web deployment
5. **Fine-tuning Potential**: Better hyperparameter optimization capabilities
6. **Industry Standard**: Widely used in production ML systems

### 📊 Current Model Performance 
> **Final model evaluation completed on test set with 18,068 samples**
> **XGBoost models optimized and evaluated - Production ready**

### 🎯 Final Test Set Results
```python
# XGBoost Classification Model (Final Evaluation):
Classification Performance:
├── Test Accuracy: 78.51%
├── Prediction Time: 0.78 seconds
├── Cross-Validation Score: 86.50%
├── Fine-tuned Parameters: {'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 200}
└── Test Set Size: 18,068 samples

# XGBoost Regression Model (Final Evaluation):
Regression Performance:
├── Test R² Score: 99.71%
├── Test MAE (Overall): 0.2459
├── Test MAE (Latitude): 0.2290
├── Test MAE (Longitude): 0.2628
├── Test RMSE: 0.4421
├── Prediction Time: 1.42 seconds
├── Cross-Validation Score: 99.71%
└── Fine-tuned Parameters: {'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 200}
```

### 📊 Detailed Classification Results by Severity
```python
# Classification Report (Test Set Performance):
Severity Level Performance:
├── Severity 1 (Light): Precision=23%, Recall=11%, F1=15% (174 samples)
├── Severity 2 (Moderate): Precision=82%, Recall=94%, F1=87% (14,051 samples)
├── Severity 3 (Serious): Precision=53%, Recall=29%, F1=37% (3,363 samples)
└── Severity 4 (Severe): Precision=30%, Recall=5%, F1=8% (480 samples)

Overall Performance:
├── Accuracy: 78.51%
├── Macro Average: Precision=47%, Recall=35%, F1=37%
└── Weighted Average: Precision=75%, Recall=79%, F1=75%
```

### 🎯 Model Details

#### Classification Model (Crash Severity)
- **Algorithm**: XGBoost Classifier
- **Target**: Severity levels (1-4)
  - 1: Light
  - 2: Moderate 
  - 3: Serious
  - 4: Severe
- **Features**: 45+ features (weather, road conditions, time factors, geographic data)
- **Hyperparameters**: Optimized with grid search

#### Regression Model (Location Prediction)
- **Algorithm**: XGBoost Regressor
- **Target**: Latitude/Longitude coordinates
- **Features**: Environmental and temporal factors
- **Use Case**: Predicting new accident locations

## 📈 Key Findings & Data Analysis

### 📊 Dataset Characteristics
- **Source**: [US Accidents Dataset (March 2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
- **Size**: 7.7M+ records (100k random samples used for training)
- **Time Range**: 2016-2023 (7 years)
- **Coverage**: 49 US States
- **Features**: 46 variables (weather, road conditions, geographic data, temporal features)
- **Sampling**: Stratified random sampling to maintain class distribution

### 📊 Explainable AI (XAI) Analyses

#### 🎯 Advanced XAI Analysis Results
```python
# Comprehensive XAI Analysis Pipeline:
XAI Analysis Components:
├── Feature Importance Analysis (XGBoost native)
├── Permutation Importance Calculation
├── Subgroup Performance Analysis
├── Day vs Night Model Comparison
├── Weekend vs Weekday Analysis
├── Regression Model Interpretability
└── Spatial Feature Analysis

# Model Performance Across Subgroups:
Day vs Night Classification:
├── Day Accuracy: 78.26%
├── Night Accuracy: 79.35%
└── Night Performance: +1.09% better

Weekend vs Weekday Classification:
├── Weekend Accuracy: 79.36%
├── Weekday Accuracy: 78.36%
└── Weekend Performance: +1.00% better

Regression Model XAI Analysis:
├── Permutation Importance: Location prediction feature ranking
├── Day/Night Comparison: Temporal location accuracy analysis
├── Weekend/Weekday Analysis: Day-type location prediction patterns
└── Spatial Feature Analysis: Geographic correlation interpretability
```

#### 🎯 Feature Importance Analysis
![Classification Feature Importance](xai_plots/classification_feature_importance.png)

**Most Important Features (Classification)**:
1. **Weather_Condition**: Weather conditions
2. **Temperature**: Temperature values
3. **Time_of_Day**: Hour of the day
4. **Road_Condition**: Road conditions
5. **State**: State information

#### 🔄 Permutation Importance
![Classification Permutation Importance](xai_plots/classification_permutation_importance.png)

**Permutation Importance Ranking**:
- Features with the most impact on model performance
- Model stability in real-world scenarios
- Importance scores validated with cross-validation

#### 🌅 Day/Night Comparison Analysis
![Day Night Comparison](xai_plots/classification_day_night_comparison.png)

**Day vs Night Accidents**:
- **Day Accuracy**: 78.26% (slightly lower performance)
- **Night Accuracy**: 79.35% (+1.09% better performance)
- **Risk Factors**: Night visibility, fatigue, alcohol consumption
- **Model Insight**: Higher accuracy at night suggests clearer decision boundaries

#### 📅 Weekend/Weekday Patterns
![Weekend Weekday Comparison](xai_plots/classification_weekend_weekday_comparison.png)

**Weekday vs Weekend**:
- **Weekend Accuracy**: 79.36% (+1.00% better performance)
- **Weekday Accuracy**: 78.36% (baseline performance)
- **Weekend Pattern**: Better model performance, possibly due to clearer traffic patterns
- **Weekdays**: More complex rush hour patterns affecting prediction accuracy
- **Friday**: Riskiest day (18% more accidents)

#### 🗺️ Regression Model XAI Analysis
![Regression Permutation Importance](xai_plots/regression_permutation_importance.png)

**Most Important Features (Location Prediction)**:
1. **State**: Geographic state information
2. **Temperature**: Environmental temperature
3. **Weather_Condition**: Weather patterns impact
4. **Time_of_Day**: Temporal location patterns
5. **County**: Fine-grained geographic data

**Location Prediction Model Interpretability**:
- **Permutation Importance**: Features critical for location accuracy
- **Geographic Features**: Spatial correlation analysis
- **Model Robustness**: Cross-validation stability testing
- **Feature Interaction**: Multivariate location dependencies

#### 🌅 Regression Day/Night Analysis
![Regression Day Night Comparison](xai_plots/regression_day_night_comparison.png)

**Location Prediction Day vs Night**:
- **Day R² Score**: Location prediction accuracy during daylight
- **Night R² Score**: Location prediction accuracy during nighttime
- **Temporal Patterns**: Time-based location prediction variations
- **Environmental Factors**: Light condition impact on location accuracy

#### 📅 Regression Weekend/Weekday Analysis
![Regression Weekend Weekday Comparison](xai_plots/regression_weekend_weekday_comparison.png)

**Location Prediction Weekend vs Weekday**:
- **Weekend R² Score**: Location accuracy during weekends
- **Weekday R² Score**: Location accuracy during weekdays
- **Traffic Patterns**: Day-type impact on location prediction
- **Spatial Distribution**: Weekend vs weekday accident location patterns

### 🗺️ Geographic & Temporal Insights

#### 🏛️ State-Based Analysis
**Highest Risk States**:
1. **California**: 19.2% total accidents
2. **Texas**: 13.8% total accidents
3. **Florida**: 11.5% total accidents
4. **New York**: 8.3% total accidents

#### 🕐 Temporal Patterns
**Critical Time Periods**:
- **Rush Hours**: 7-9 AM (23%), 4-6 PM (28%)
- **Dangerous Hours**: 02:00-04:00 (most severe)
- **Seasonal**: 34% more severe in winter months

#### 🌤️ Weather Impact Analysis
**Weather Distribution**:
- **Clear**: 73% (low severity)
- **Rain**: 18% (moderate severity)
- **Snow**: 4% (high severity)
- **Fog**: 3% (highest severity)
- **Severe Weather**: 2% (critical severity)

### 🛣️ Road Condition Analysis

#### 🚦 Road Type Impact
**Risk by Road Type**:
- **Highway**: Lower frequency, higher severity
- **Urban Roads**: High frequency, low severity
- **Rural Roads**: Medium frequency, high fatality rate
- **Interstate**: I-95, I-75, I-10 most risky

#### 🚧 Traffic Features
**Traffic Conditions**:
- **Traffic_Signal**: 45% of accidents near traffic lights
- **Stop_Sign**: 23% near stop signs
- **Speed_Limit**: 2.3x more severe at >65 mph
- **Construction**: 67% more risky in construction zones

### 🎯 Model Insights & Business Value

#### 📈 Prediction Accuracy by Conditions
**Prediction Success by Conditions**:
- **Clear Weather**: 89.2% accuracy
- **Rain**: 76.8% accuracy
- **Snow**: 71.3% accuracy
- **Fog**: 68.9% accuracy

#### 💼 Business Applications
**Practical Use Cases**:
1. **Insurance**: Risk assessment and premium calculation
2. **Traffic Management**: Proactive traffic management
3. **Emergency Services**: Resource optimization
4. **Urban Planning**: Safe road design
5. **Fleet Management**: Route optimization

## 🔧 Technical Implementation

### 📊 Data Preprocessing Pipeline
```python
# Data preprocessing steps:
1. Missing Value Imputation (KNN Imputer)
2. Categorical Encoding (Target Encoding)
3. Feature Engineering (Temporal, Geographic)
4. Outlier Detection (IQR Method)
5. Feature Scaling (StandardScaler)
6. Class Balancing (SMOTE)
```

### 🧠 Model Training Pipeline
```python
# Model training process:
1. Train/Validation/Test Split (60/20/20)
2. Hyperparameter Tuning (GridSearchCV)
3. Cross-Validation (5-fold)
4. Model Ensemble (Voting Classifier)
5. Feature Selection (RFE)
6. Model Validation (Out-of-sample testing)
```

### 🔍 Explainable AI (XAI) Implementation
```python
# XAI Techniques:
1. ✅ SHAP (SHapley Additive exPlanations)
   - Global feature importance
   - Local explanations
   - Partial dependence plots
   - Interaction effects

2. ✅ LIME (Local Interpretable Model-agnostic Explanations)
   - Instance-level explanations
   - Feature contribution analysis
   - Counterfactual explanations

3. ✅ Permutation Importance
   - Model-agnostic importance
   - Feature interaction detection
   - Robustness testing

4. ✅ Advanced Subgroup Analysis
   - Day vs Night performance comparison
   - Weekend vs Weekday analysis
   - Temporal pattern interpretability
   - Regression model feature importance
   - Spatial correlation analysis
   - Location prediction XAI

5. ✅ Visualization Tools
   - Feature importance plots
   - Partial dependence plots
   - SHAP summary plots
   - Individual prediction explanations
   - Subgroup comparison visualizations
   - Regression model interpretability plots
   - Spatial correlation analysis
```

## 🚀 Quick Start & Installation

### 📋 Prerequisites
- Python 3.13.5 (recommended)
- Git
- 16GB+ RAM (for large dataset processing)
- 40GB+ free disk space

### 1. 📥 Repository Setup
```bash
git clone https://github.com/anilmetin0/us-traffic-crash-ml.git

# Clone the repository
git clone https://github.com/anilmetin0/us-traffic-crash-ml.git
cd us-traffic-crash-ml

# Or download as ZIP and extract
```

### 2. 🐍 Python Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. 📊 Data Analysis & Model Training
```bash
# Start Jupyter Notebook
jupyter notebook

# Open the main notebook
# US_Traffic_Crash_ML.ipynb

# Run all cells sequentially
```

### 4. 🌐 Streamlit Web Application

#### Local Development:
```bash
# Go to deployment folder
cd streamlit_deployment

# Install Streamlit dependencies
pip install -r requirements.txt

# Start the application
streamlit run app.py

# Open in browser: http://localhost:8501
```

#### Production Deployment:

##### 🚀 Streamlit Cloud (Recommended):
1. Push repository to GitHub
2. Create [Streamlit Cloud](https://streamlit.io/cloud) account
3. Connect repository
4. Select `streamlit_deployment/app.py` as main file
5. Deploy

##### 🐳 Docker Deployment:
```bash
# Create Dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY streamlit_deployment/ .

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

```bash
# Docker build and run
docker build -t traffic-crash-app .
docker run -p 8501:8501 traffic-crash-app
```

### 5. 🎯 Usage Examples

#### Web Application Usage:
1. **Accident Location**: Select state and county
2. **Time Information**: Enter date and time
3. **Weather Conditions**: Current weather conditions
4. **Road Conditions**: Traffic and road conditions
5. **Get Prediction**: Accident severity and risk score

#### API Usage (For Developers):
```python
import pickle
import pandas as pd

# Load model and preprocessor
with open('streamlit_deployment/severity_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open('streamlit_deployment/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Prepare sample data
data = pd.DataFrame({
    'Temperature': [65.0],
    'Humidity': [78.0],
    'Pressure': [29.85],
    'Weather_Condition': ['Clear'],
    'State': ['CA'],
    # ... other features
})

# Make prediction
processed_data = preprocessor.transform(data)
prediction = model.predict(processed_data)
probability = model.predict_proba(processed_data)
```

## 🌐 Live Demo & Examples

### 📱 Web Application Features:
- **Interactive Interface**: User-friendly interface
- **Real-time Predictions**: Instant prediction results
- **Risk Visualization**: Visual risk assessment
- **Input Validation**: Comprehensive data validation
- **Mobile Responsive**: Mobile-compatible design
- **County Selection**: Easy selection with dropdown

### 🎥 Demo Screenshots:
```
[Home Page] → [Data Input] → [Prediction Results] → [Risk Analysis]
```

### 🔗 Live Application:
> **Demo Link**: [US Traffic Crash ML](https://us-traffic-crash-ml.streamlit.app/)

## 📊 Dataset & Files Description

### 📄 Core Files:
| File | Size | Description |
|------|-------|----------|
| `US_Accidents_March23.csv` | ~3GB | Main dataset (7.7M records) - [Download from Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) |
| `US_Traffic_Crash_ML.ipynb` | ~82KB | Jupyter notebook (full analysis) |
| `requirements.txt` | ~1KB | Development environment dependencies |

### 🌐 Deployment Files:
| File | Size | Description |
|------|-------|----------|
| `app.py` | ~13KB | Streamlit web application |
| `severity_classifier.pkl` | ~2.8MB | XGBoost classification model |
| `location_regressor.pkl` | ~1MB | XGBoost regression model |
| `preprocessor.pkl` | ~23KB | Scikit-learn preprocessing pipeline |
| `model_metadata.json` | ~3KB | Model performance metrics |
| `feature_names.pkl` | ~33KB | Feature names (post-processing) |
| `unique_counties.txt` | ~18KB | US state/county list |
| `requirements.txt` | ~1KB | Production environment dependencies |

### 📈 Analysis Outputs:
| Folder/File | Description |
|-------------|----------|
| `xai_plots/` | **Explainable AI visualizations (Advanced XAI Analysis)** |
| `├── classification_feature_importance.png` | Classification feature importance ranking |
| `├── classification_permutation_importance.png` | Permutation importance analysis |
| `├── classification_day_night_comparison.png` | Day/night accuracy comparison |
| `├── classification_weekend_weekday_comparison.png` | Weekend/weekday comparison |
| `├── regression_permutation_importance.png` | Regression model importance analysis |
| `├── regression_day_night_comparison.png` | Location prediction day/night analysis |
| `└── regression_weekend_weekday_comparison.png` | Regression weekend/weekday analysis |

## 🔧 Technical Features & Architecture

### 🛠️ Data Processing Pipeline
```python
# Data Processing Steps:
1. ✅ Data Loading & Initial Exploration
   - Loading 7.7M+ records, sampling 100k for training
   - Stratified random sampling to maintain class distribution
   - Missing data analysis
   - Data type optimization

2. ✅ Data Cleaning & Preprocessing
   - Missing value imputation (KNN, Median, Mode)
   - Outlier detection (IQR, Z-score)
   - Data type conversions
   - Duplicate removal

3. ✅ Feature Engineering
   - Temporal features (hour, day, month, season)
   - Geographic features (region, urban/rural)
   - Weather condition grouping
   - Distance calculations

4. ✅ Categorical Encoding
   - Target encoding for high-cardinality features
   - One-hot encoding for binary features
   - Ordinal encoding for ordered categories

5. ✅ Feature Selection
   - Correlation analysis
   - Mutual information scoring
   - Recursive feature elimination (RFE)
   - SHAP-based feature importance
```

### 🧠 Machine Learning Pipeline
```python
# ML Workflow:
1. ✅ Data Splitting
   - Train: 60% (~60k records from 100k sample)
   - Validation: 20% (~20k records from 100k sample)
   - Test: 20% (~20k records from 100k sample)
   - Stratified sampling to maintain class distribution

2. ✅ Class Imbalance Handling
   - SMOTE (Synthetic Minority Oversampling)
   - Random undersampling
3. ✅ Model Training & Tuning
   - Grid search hyperparameter tuning
   - 3-fold cross-validation for optimization
   - XGBoost Classifier: 2707.70 seconds tuning time
   - XGBoost Regressor: 349.77 seconds tuning time
   - Early stopping mechanisms
   - Regularization techniques

4. ✅ Model Evaluation
   - Multiple metrics (Accuracy, F1, AUC-ROC)
   - Confusion matrices
   - Classification reports
   - Cross-validation scores

5. ✅ Model Validation
   - Out-of-sample testing
   - Temporal validation
   - Geographic validation
   - Bootstrap sampling
```

### 🔍 Explainable AI (XAI) Implementation
```python
# XAI Techniques:
1. ✅ SHAP (SHapley Additive exPlanations)
   - Global feature importance
   - Local explanations
   - Partial dependence plots
   - Interaction effects

2. ✅ LIME (Local Interpretable Model-agnostic Explanations)
   - Instance-level explanations
   - Feature contribution analysis
   - Counterfactual explanations

3. ✅ Permutation Importance
   - Model-agnostic importance
   - Feature interaction detection
   - Robustness testing

4. ✅ Visualization Tools
   - Feature importance plots
   - Partial dependence plots
   - SHAP summary plots
   - Individual prediction explanations
```

### 🌐 Web Application Architecture
```python
# Streamlit App Features:
1. ✅ Interactive UI Components
   - Dropdown menus (counties, states)
   - Slider inputs (temperature, humidity)
   - Date/time pickers
   - Radio buttons (weather conditions)

2. ✅ Real-time Prediction Engine
   - Model loading and caching
   - Input validation
   - Preprocessing pipeline
   - Prediction generation

3. ✅ Risk Assessment Dashboard
   - Severity level visualization
   - Risk score calculation
   - Probability distributions
   - Confidence intervals

4. ✅ User Experience Features
   - Input validation with feedback
   - Loading indicators
   - Error handling
   - Mobile responsive design
   - Developer attribution footer
```

## 📊 Model Performance Metrics

### 🎯 Current Model Performance
> **Final model evaluation completed on test set with 18,068 samples**
> **Model trained on 100k randomly sampled records from 7.7M+ dataset**

### 📈 Detailed Performance Analysis
```python
# Model Evaluation Metrics:
Classification Model (XGBoost):
├── Test Accuracy: 78.51%
├── Precision (Weighted): 75%
├── Recall (Weighted): 79%
├── F1-Score (Weighted): 75%
├── CV Score: 86.50%
└── Prediction Time: 0.78 seconds

Regression Model (XGBoost):
├── R² Score: 99.71%
├── RMSE: 0.4421
├── MAE (Overall): 0.2459
├── MAE (Latitude): 0.2290
├── MAE (Longitude): 0.2628
└── Prediction Time: 1.42 seconds
```

### 🔄 Cross-Validation Results
```python
# Production Model Performance (Post Fine-tuning):
Classification:
├── CV Score: 86.50%
├── Hyperparameters: {'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 200}
├── Test Accuracy: 78.51%
└── Prediction Speed: 0.78 seconds

Regression:
├── CV Score: 99.71%
├── Hyperparameters: {'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 200}
├── Test R²: 99.71%
└── Prediction Speed: 1.42 seconds
```

### 📊 Algorithm Comparison Summary
```python
# Classification Cross-Validation Results:
Algorithm Performance (F1-Macro):
├── Random Forest: 95.22% (Best CV Score)
├── HistGradientBoosting: 88.69%
├── XGBoost: 87.60% (Selected for Production)
├── KNN: 81.96%
├── Logistic Regression: 68.34%
└── Naive Bayes: 45.78%

# Regression Cross-Validation Results:
Algorithm Performance (R²):
├── Random Forest: 99.91% (Best CV Score)
├── Linear Regression: 99.77%
├── Ridge: 99.77%
├── XGBoost: 99.57% (Selected for Production)
├── HistGradientBoosting: 99.50%
└── KNN: 91.53%
```

### 🎯 Model Fine-Tuning Results
```python
# Grid Search Hyperparameter Optimization:
XGBoost Classifier Fine-Tuning:
├── Grid Search Duration: 2707.70 seconds (~45 minutes)
├── Total Fits: 24 (3 folds × 8 parameter combinations)
├── Best Parameters: {'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 200}
├── Best CV Score (F1-Macro): 86.50%
└── Optimization Method: 3-fold cross-validation

XGBoost Regressor Fine-Tuning:
├── Grid Search Duration: 349.77 seconds (~6 minutes)
├── Total Fits: 24 (3 folds × 8 parameter combinations)
├── Best Parameters: {'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 200}
├── Best CV Score (R²): 99.71%
└── Optimization Method: 3-fold cross-validation
```

### 📊 Classification Model - Detailed Results
```python
# Classification Report by Severity Level:
Severity 1 (Light):
├── Precision: 23%
├── Recall: 11%
├── F1-Score: 15%
└── Support: 174 samples

Severity 2 (Moderate):
├── Precision: 82%
├── Recall: 94%
├── F1-Score: 87%
└── Support: 14,051 samples

Severity 3 (Serious):
├── Precision: 53%
├── Recall: 29%
├── F1-Score: 37%
└── Support: 3,363 samples

Severity 4 (Severe):
├── Precision: 30%
├── Recall: 5%
├── F1-Score: 8%
└── Support: 480 samples
```

### 📈 Model Performance Insights
- **High Overall Accuracy**: 78.51% on test set with strong cross-validation score of 86.50%
- **Efficient Training**: Model trained on 100k stratified random samples from 7.7M+ dataset
- **Comprehensive Evaluation**: 6 algorithms tested for both classification and regression
- **Optimized Selection**: XGBoost chosen for production despite Random Forest's higher CV scores
- **Optimized Hyperparameters**: 45-minute grid search optimization for classification, 6-minute for regression
- **Class Imbalance Impact**: Best performance on Severity 2 (moderate) accidents (87% F1-score)
- **Regression Excellence**: 99.71% R² score for location prediction with low error rates
- **Fast Prediction**: Sub-second prediction times suitable for real-time applications
- **Robust Location Prediction**: MAE of 0.2459 degrees (~27km average error)
- **Scalable Approach**: Demonstrates effective performance with reduced dataset size

## 📋 Dependencies & Requirements

### 🐍 Development Environment
```txt
# requirements.txt (Main project)
Python==3.13.5
pandas==2.3.1
numpy==2.3.1
scikit-learn==1.6.1
xgboost==3.0.2
matplotlib==3.10.3
seaborn==0.13.2
plotly==6.2.0
shap==0.48.0
lime==0.2.0.1
jupyter==1.0.0
notebook==7.0.6
ipykernel==6.29.0
```

### 🌐 Production Environment
```txt
# streamlit_deployment/requirements.txt
streamlit==1.47.0
pandas==2.3.1
numpy==2.3.1
scikit-learn==1.6.1
xgboost==3.0.2
plotly==6.2.0
pickle-protocol==5
```

### 📦 Package Explanations
| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | 2.3.1 | Data manipulation and analysis |
| `numpy` | 2.3.1 | Numerical computations |
| `scikit-learn` | 1.6.1 | ML algorithms and preprocessing |
| `xgboost` | 3.0.2 | Main ML model |
| `streamlit` | 1.47.0 | Web application framework |
| `shap` | 0.48.0 | Model explainability |
| `plotly` | 6.2.0 | Interactive visualization |
| `seaborn` | 0.13.2 | Statistical visualization |
| `matplotlib` | 3.10.3 | Basic plotting |

## 🚀 Deployment Options & Cloud Services

### ☁️ Cloud Deployment Options

#### 1. 🌟 Streamlit Cloud (Recommended)
```bash
# Advantages:
✅ Free hosting
✅ GitHub integration
✅ Automatic deployment
✅ SSL certificate
✅ Custom domain support

# Setup:
1. Push to GitHub
2. Create Streamlit Cloud account
3. Connect repository
4. Select streamlit_deployment/app.py
5. Deploy
```

#### 2. 🐳 Docker + Cloud Run
```dockerfile
# Dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY streamlit_deployment/ .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
```

## 🤝 Contributing & Development

### 🔄 Development Workflow
```bash
# Development setup:
1. Fork repository
2. Create feature branch: git checkout -b feature/new-feature
3. Make changes and test
4. Update documentation
5. Commit changes: git commit -am 'Add new feature'
6. Push to branch: git push origin feature/new-feature
7. Create Pull Request
```

### 📋 Development Guidelines
```python
# Code Standards:
✅ PEP 8 compliance
✅ Type hints for functions
✅ Docstrings for all functions
✅ Unit tests for new features
✅ Performance optimization
✅ Security best practices
```

### 📚 Documentation
```markdown
# Documentation structure:
├── README.md (this file)
├── docs/
│   ├── API.md
│   ├── DEPLOYMENT.md
│   ├── CONTRIBUTING.md
│   └── CHANGELOG.md
└── examples/
    ├── usage_examples.py
    └── api_examples.py
```

## 🔍 Troubleshooting & FAQ

### ❓ Common Issues

#### 1. Memory Issues
```bash
# Problem: "MemoryError during data loading"
# Solution:
- Increase system RAM or use cloud instance
- Use pandas chunking: pd.read_csv(..., chunksize=10000)
- Enable memory optimization in notebook
```

#### 2. Model Loading Issues
```bash
# Problem: "Model file not found"
# Solution:
- Check file paths in streamlit_deployment/
- Ensure all .pkl files are present
- Re-run notebook Cell 14 to regenerate models
```

#### 3. Streamlit Deployment Issues
```bash
# Problem: "Package import errors"
# Solution:
- Check requirements.txt versions
- Use virtual environment
- Update pip: pip install --upgrade pip
```

### 🆘 Support & Help
```markdown
# Getting Help:
🐛 Issues: GitHub Issues tab
📖 Documentation: README.md
💬 Discussions: GitHub Discussions
```

## 📊 Performance Optimization

### 🚀 Speed Optimizations
```python
# Performance Tips:
1. ✅ Use vectorized operations (NumPy/Pandas)
2. ✅ Implement caching (@st.cache_data)
3. ✅ Optimize data types (category vs object)
4. ✅ Use efficient algorithms (XGBoost vs sklearn)
5. ✅ Batch processing for large datasets
```

### 💾 Memory Optimizations
```python
# Memory Management:
1. ✅ Delete unused variables: del variable
2. ✅ Use generators for large datasets
3. ✅ Implement garbage collection: gc.collect()
4. ✅ Optimize DataFrame dtypes
5. ✅ Use sparse matrices when applicable
```

## 🔐 Security & Privacy

### 🛡️ Security Measures
```python
# Security implementations:
1. ✅ Input validation and sanitization
2. ✅ No sensitive data in repository
3. ✅ Environment variables for secrets
4. ✅ HTTPS-only deployment
5. ✅ Regular dependency updates
```

### 🔒 Privacy Considerations
```markdown
# Privacy compliance:
✅ No personal data collection
✅ Anonymous prediction service
✅ Local processing option
✅ GDPR compliant
✅ Data retention policies
```

## 📚 References & Citations

### 📖 Academic References
```bibtex
@article{traffic_prediction_2025,
  title={US Traffic Crash Prediction Using Machine Learning},
  author={Anıl Metin, Sinem Söner},
  journal={CMPE442 Course Project},
  year={2025},
  institution=TED University
}
```

### 🔗 Dataset Citations
```bibtex
@dataset{us_accidents_2023,
  title={US Accidents Dataset},
  author={Sobhan Moosavi},
  year={2023},
  publisher={Kaggle},
  url={https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents}
}
```

### 📋 Technical References
```markdown
# Key References:
1. XGBoost Documentation: https://xgboost.readthedocs.io/
2. Streamlit Documentation: https://docs.streamlit.io/
3. SHAP Documentation: https://shap.readthedocs.io/
4. Scikit-learn Documentation: https://scikit-learn.org/
5. Pandas Documentation: https://pandas.pydata.org/
```

## 📈 Future Enhancements

### 🔮 Roadmap
```markdown
# Version 2.0 Planned Features:
🎯 Real-time traffic data integration
🎯 Advanced ensemble methods
🎯 Deep learning models (LSTM, CNN)
🎯 Mobile application
🎯 API service development
🎯 Multi-language support
```

### 🆕 Potential Improvements
```python
# Enhancement Ideas:
1. 🔄 Real-time data streaming
2. 🌍 International dataset support
3. 📱 Mobile-first design
4. 🤖 Automated retraining pipeline
5. 🔍 Advanced explainability features
6. 📊 Interactive dashboard
7. 🚨 Alert system integration
```

## 📄 License & Legal

### 📜 License Information
```markdown
# MIT License
Copyright (c) 2025 Anıl Metin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### ⚖️ Legal Disclaimers
```markdown
# Important Disclaimers:
⚠️ This project is for educational and research purposes only
⚠️ Predictions should not be used for critical safety decisions
⚠️ Always consult official traffic safety authorities
⚠️ Dataset may contain biases and limitations
⚠️ Model performance may vary across different conditions
```

## 🙏 Acknowledgments

- **Dataset**: Sobhan Moosavi for the US Accidents dataset
- **Libraries**: Thanks to the open-source community
- **Inspiration**: Traffic safety research community

## 📞 Support

If you have any questions or issues:
1. Check the [Issues](https://github.com/anilmetin0/us-traffic-crash-ml/issues) page
2. Create a new issue with detailed description
3. Contact via email for urgent matters

---

⭐ **If this project helped you, please give it a star!** ⭐
