# 🛒 E-Commerce Sales Data Analyzer & Predictive AI

An end-to-end Data Science and Machine Learning project that ingests raw retail transactional data, performs rigorous data cleaning, extracts actionable business intelligence through Exploratory Data Analysis (EDA), and deploys an optimized Random Forest machine learning model to predict sales performance.

---

## 📌 Project Overview
In modern e-commerce, understanding revenue drivers and accurately forecasting sales is critical for supply chain efficiency and marketing optimization. This project provides a complete data pipeline that automates data wrangling, visualizes temporal and categorical sales patterns, and predicts order values using supervised machine learning.

## 🎯 Problem Statement
- **Business Challenge:** How can an e-commerce platform identify revenue-driving patterns and reliably predict order sales values based on customer and product attributes?
- **Technical Goal:** Build an end-to-end regression pipeline that cleans messy data, engineers temporal features, trains a robust Random Forest model, and achieves high predictive accuracy.

## 📊 Dataset
- **Source:** Kaggle "Superstore Sales Dataset"
- **Features Used:** `Category`, `Sub-Category`, `Segment`, `Region`, `Order Date`, `Sales`

## 🛠️ Technologies & Libraries Used
- **Python 3.12**
- **Pandas & NumPy:** Data wrangling, preprocessing, and missing value analysis.
- **Matplotlib & Seaborn:** Exploratory Data Analysis (EDA) and business chart visualization.
- **Scikit-learn:** Machine learning modeling (`RandomForestRegressor`), data scaling (`StandardScaler`), train-test splitting, and hyperparameter tuning (`GridSearchCV`).
- **Joblib:** Model serialization and deployment.
- **Google Colab:** Cloud-based development environment.

---

## 🔍 Data Preprocessing & Pipeline
1. **Data Inspection:** Evaluated column structures, data types, and null value counts via `.info()` and `.describe()`.
2. **Data Cleaning:** Handled missing values, checked for duplicate rows, and verified logical constraints (e.g., zero or negative sales).
3. **Datetime Parsing:** Converted international string dates (`DD/MM/YYYY`) into standard `datetime64` objects using `pd.to_datetime()`.
4. **Feature Engineering:** Extracted temporal components like `Month-Year` periods and `Day of Week` names.
5. **One-Hot Encoding:** Converted categorical text features into machine-readable binary columns (1s and 0s) using `pd.get_dummies()`.
6. **Feature Scaling:** Standardized feature distributions using `StandardScaler` (fitted exclusively on the training set to prevent data leakage).

---

## 📈 Exploratory Data Analysis (EDA) & Key Findings
- **Category Dominance:** Visualized cumulative sales by product category, highlighting which departments drive the majority of revenue.
- **Seasonal Trends:** Built a time-series line chart tracking monthly sales over time, revealing retail spikes during late-year holiday shopping seasons.
- **Shopping Behavior:** Analyzed revenue distribution across days of the week, confirming peak customer activity.

---

## 🤖 Machine Learning Model Development
- **Algorithm:** Random Forest Regressor (ensemble of 100 decision trees).
- **Target Variable (`y`):** `Sales` dollar amount.
- **Evaluation Strategy:** 80/20 Train-Test Split combined with 5-Fold Cross-Validation.
- **Hyperparameter Optimization:** Utilized `GridSearchCV` to tune `n_estimators`, `max_depth`, and `min_samples_split`.

---

## 📊 Model Evaluation & Results
- **Metrics Tracked:**
  - **Mean Absolute Error (MAE):** Quantifies average dollar error.
  - **Root Mean Squared Error (RMSE):** Heavily penalizes large variance errors.
  - **R-squared ($R^2$):** Measures overall variance captured by the model.
- **Feature Importance:** Extracted feature importance weights, showing that product categorization heavily dominates sales prediction accuracy.

---

## 🚀 How to Run the Project
1. Open the project notebook in **Google Colab**.
2. Mount your Google Drive or upload the Superstore Sales dataset into your workspace.
3. Run cells sequentially to execute data cleaning, EDA, and model training.
4. Load the serialized model using `joblib` for future inference:
   ```python
   import joblib
   loaded_model = joblib.load('ecommerce_sales_model.pkl')
