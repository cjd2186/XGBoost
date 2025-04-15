# XGBoost
Competition URL:
https://www.kaggle.com/competitions/coms-4771-fall-2024-regression-competition/overview 

---

# 📊 Time Duration Prediction with XGBoost and Feature Engineering

This project focuses on predicting durations using machine learning, with a strong emphasis on feature preprocessing, correlation analysis, and advanced model tuning using **XGBoost** and **Bayesian optimization**. It also experiments with **linear regression** and **Prophet** time series forecasting.

---

## 🔧 Installation

```bash
pip install feature-engine xgboost seaborn prophet scikit-optimize
```

---

## 📁 Project Structure

- `train_examples.csv`: Training features
- `train_labels.csv`: Duration labels
- `test_examples.csv`: Test features

---

## 📈 Workflow Overview

### 1. **Data Preprocessing**
- Drop highly correlated features (`feature_4`, `feature_5`, `feature_6`, `feature_7`, `feature_10`).
- Parse `feature_0` (datetime string) into cyclic time components (`month`, `day`, `hour`, `minute`, `second`).
- Merge cyclical time features with the rest of the dataset.

### 2. **Exploratory Data Analysis**
- Generate heatmaps of feature correlations.
- Visualize feature distributions with seaborn (optional).

### 3. **Modeling Approaches**

#### ✅ Linear Regression
- Baseline model using `sklearn.LinearRegression`
- Evaluated using MAE

#### 🌲 XGBoost Regression
- Trained using `xgboost.DMatrix` and `XGBRegressor`
- Hyperparameter tuning with `BayesSearchCV` (`skopt`)
- Early stopping and custom loss function support

#### 🔄 Pipeline Integration
- Built a `Pipeline` with `ColumnTransformer` for consistent preprocessing
- Combined cyclical datetime encoding with modeling for reproducibility

#### 🔮 Prophet Forecasting (Experimental)
- Transformed `feature_0` to a time series (`ds`)
- Fit Prophet for trend/seasonality inspection

---

## 📊 Evaluation Metrics
- **MAE** (Mean Absolute Error)
- **RMSE** (for tuned models)

---

## 📤 Outputs
- Final predictions saved to `y_pred_12.csv`
- Predictions are integer-cast durations with proper indexing

---

## 🛠️ Key Tools & Libraries
- `XGBoost`, `scikit-learn`, `BayesSearchCV` (hyperparam tuning)
- `Feature-engine` for cyclical encoding
- `Prophet` for time-series modeling
- `seaborn` and `matplotlib` for visualization

---

## 🚀 Next Steps
- Tune feature selection further
- Explore additional time series models
- Integrate more robust cross-validation strategies

---

