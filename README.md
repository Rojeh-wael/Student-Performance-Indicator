# Student Performance Indicator - End-to-End ML Project

## Project Overview

This is a comprehensive end-to-end machine learning project designed to predict student performance scores (math, reading, and writing) based on demographic and educational factors. The project implements a complete MLOps pipeline with data ingestion, transformation, model training, and evaluation.

### Key Features
- **Data Pipeline**: Automated data ingestion and preprocessing
- **Feature Engineering**: OneHot encoding for categorical variables and standardization for numerical features
- **Multi-Model Training**: Grid search hyperparameter tuning across 8 different regression models
- **Model Evaluation**: Cross-validated model comparison using R² scores
- **Artifact Management**: Saves preprocessor and best-trained model as pickle files

---

## Project Architecture

```
Student Performance Indicator Project/
├── artifacts/                          # Output directory
│   ├── data.csv                       # Raw dataset
│   ├── train.csv                      # Training set (80%)
│   ├── test.csv                       # Test set (20%)
│   ├── preprocessor.pkl               # Fitted preprocessing pipeline
│   └── model.pkl                      # Best trained model
├── logs/                              # Application logs
├── notebook/                          # Jupyter notebooks for EDA
│   ├── 1 . EDA STUDENT PERFORMANCE .ipynb
│   ├── 2. MODEL TRAINING.ipynb
│   └── data/StudentsPerformance.csv
├── src/                               # Main source code
│   ├── __init__.py
│   ├── exception.py                   # Custom exception handling
│   ├── logger.py                      # Logging configuration
│   ├── utils.py                       # Utility functions (evaluate_models, save_object)
│   └── components/
│       ├── data_ingestion.py          # Data loading and train-test split
│       ├── data_transformation.py     # Feature preprocessing & scaling
│       └── model_trainer.py           # Model training & hyperparameter tuning
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package setup
└── README.md                          # This file
```

---

## Dataset

### Source
- **File**: `notebook/data/StudentsPerformance.csv`
- **Samples**: 1,000 student records
- **Train/Test Split**: 80% train, 20% test (random_state=42)

### Features
**Categorical Features** (5):
- `gender`: male/female
- `race/ethnicity`: groups A–E
- `parental level of education`: education level of parents
- `lunch`: standard/free reduced
- `test preparation course`: completed/none

**Target Variables** (3):
- `math score`: 0–100
- `reading score`: 0–100
- `writing score`: 0–100

---

## Pipeline Overview

### 1. Data Ingestion (`src/components/data_ingestion.py`)
- Loads CSV from `notebook/data/StudentsPerformance.csv`
- Splits into train (80%) and test (20%) sets
- Saves splits to `artifacts/train.csv` and `artifacts/test.csv`
- **Returns**: Paths to train and test CSV files

### 2. Data Transformation (`src/components/data_transformation.py`)
- **Preprocessing Pipeline**:
  - **Categorical Features**: SimpleImputer (most_frequent) → OneHotEncoder → StandardScaler
  - **Numerical Features**: None (targets were previously being incorrectly included)
- **Output**: 
  - Transforms features and concatenates with all 3 target variables
  - Returns numpy arrays: `train_array` (800, 20) and `test_array` (200, 20)
  - Saves fitted preprocessor to `artifacts/preprocessor.pkl`

### 3. Model Training (`src/components/model_trainer.py`)
- **Models Trained** (8 total):
  1. Random Forest Regressor
  2. Decision Tree Regressor
  3. Gradient Boosting Regressor
  4. Linear Regression
  5. XGBRegressor
  6. CatBoost Regressor
  7. AdaBoost Regressor
  8. K-Neighbors Regressor

- **Hyperparameter Tuning**: GridSearchCV with cv=5 for each model
- **Evaluation Metric**: R² Score (coefficient of determination)
- **Output**: 
  - Best model saved to `artifacts/model.pkl`
  - Returns R² score of best model on test set

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip or conda

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies**:
- pandas, numpy — Data manipulation
- scikit-learn — ML algorithms and preprocessing
- catboost, xgboost — Gradient boosting libraries
- matplotlib, seaborn — Visualization (for notebooks)
- dill — Advanced pickling for model serialization

### Step 2: Project Structure
Ensure all directories exist:
```bash
mkdir -p artifacts logs notebook/data src/components
```

### Step 3: Data File
Place your dataset at: `notebook/data/StudentsPerformance.csv`

---

## Usage

### Option 1: Run Full Pipeline
Execute the main script in `src/components/data_ingestion.py`:

```bash
python -m src.components.data_ingestion
```

This will:
1. Load and split the dataset
2. Transform features (OneHot encode, scale)
3. Train 8 models with hyperparameter tuning
4. Select and save the best model
5. Print the best model's R² score

### Option 2: Step-by-Step Execution (Python)
```python
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Step 1: Data Ingestion
obj = DataIngestion()
train_path, test_path = obj.initiate_data_ingestion()

# Step 2: Data Transformation
data_transformation = DataTransformation()
train_array, test_array = data_transformation.initiate_data_transformation(train_path, test_path)

# Step 3: Model Training
model_trainer = ModelTrainer()
r2_score = model_trainer.initiate_model_trainer(train_array, test_array)
print(f"Best Model R² Score: {r2_score}")
```

### Option 3: Use Jupyter Notebooks
Notebooks for exploratory data analysis (EDA) are available in `notebook/`:
- `1 . EDA STUDENT PERFORMANCE .ipynb` — Data exploration and visualization
- `2. MODEL TRAINING.ipynb` — Model training pipeline

---

## Bug Fixes & Improvements Applied

### Issue 1: ImportError on r2_score
**Problem**: `ImportError: cannot import name 'r2_score' from 'sklearn.base'`
**Root Cause**: r2_score is in `sklearn.metrics`, not `sklearn.base`
**Fix**: Updated `src/utils.py` line 4 to import from correct module
```python
# Before: from sklearn.base import r2_score
# After:
from sklearn.metrics import r2_score
```

### Issue 2: ValueError - Too Many Values to Unpack
**Problem**: `ValueError: too many values to unpack (expected 2)`
**Root Cause**: `data_transformation.initiate_data_transformation()` returned 4 values (X_train, y_train, X_test, y_test) but caller expected 2
**Fix**: Modified `data_transformation.py` to:
- Fit the preprocessor on training features
- Transform both train and test features
- Concatenate features with target variables into numpy arrays
- Return exactly 2 arrays: `train_array` and `test_array`

### Issue 3: Column Not Found in ColumnTransformer
**Problem**: `ValueError: A given column is not a column of the dataframe`
**Root Cause**: Preprocessor tried to transform 'math score', 'reading score', 'writing score' columns, but these were dropped (they're targets)
**Fix**: Removed target column names from `numerical_columns` list in `get_data_transformer_object()`
```python
# Before: numerical_columns = ['math score', 'reading score', 'writing score']
# After:
numerical_columns = []
```

### Issue 4: Sparse Matrix Error
**Problem**: `ValueError: Cannot center sparse matrices: pass 'with_mean=False' instead`
**Root Cause**: OneHotEncoder outputs sparse matrices by default in newer scikit-learn versions; StandardScaler can't center sparse matrices
**Fix**: Added `sparse_output=False` to OneHotEncoder in categorical pipeline
```python
OneHotEncoder(handle_unknown='ignore', sparse_output=False)
```

### Issue 5: KeyError in evaluate_models
**Problem**: `KeyError: 'Decision Tree'` when accessing hyperparameter dict
**Root Cause**: Function incorrectly reassigned the `param_distributions` parameter inside loop, overwriting the entire dict
**Fix**: Rewrote `evaluate_models()` in `src/utils.py` to:
- Iterate using `.items()` instead of range + indexing
- Use `.get()` for safe dictionary lookup
- Actually use GridSearchCV with the retrieved parameters
- Store and return best estimator results

**Before** (broken):
```python
for i in range(len(models)):
    model = list(models.values())[i]
    param_distributions = param_distributions[list(models.keys())[i]]  # BUG: overwrites param dict!
    gscv = GridSearchCV(model, param_distributions, cv=5, n_jobs=-1)
    model.fit(X_train, y_train)  # Uses model, not gscv - GridSearchCV never used!
```

**After** (fixed):
```python
for model_name, model in models.items():
    params = param_distributions.get(model_name, {})
    if params:
        gscv = GridSearchCV(model, params, cv=5, n_jobs=-1)
        gscv.fit(X_train, y_train)
        best_model = gscv.best_estimator_
    else:
        model.fit(X_train, y_train)
        best_model = model
    y_test_pred = best_model.predict(X_test)
    test_model_score = r2_score(y_test, y_test_pred)
    report[model_name] = test_model_score
```

---

## Model Performance

After training and hyperparameter tuning on the student performance dataset:

| Model | Purpose |
|-------|---------|
| Best Model | Selected based on highest R² score on test set |
| Saved Location | `artifacts/model.pkl` |
| Preprocessor | `artifacts/preprocessor.pkl` |

**Training Time**: ~5–10 minutes (GridSearchCV with cv=5 across 8 models)

---

## Key Components

### Exception Handling (`src/exception.py`)
Custom exception class with detailed error messages and line-number tracking for debugging

### Logging (`src/logger.py`)
Configured logging for tracking pipeline execution and debugging

### Utilities (`src/utils.py`)
- `save_object()`: Pickle serialization for models and preprocessors
- `evaluate_models()`: Model training and evaluation with GridSearchCV

---

## Configuration & Hyperparameters

### GridSearchCV Parameters (cv=5)
Located in `src/components/model_trainer.py`:

```python
param_distributions = {
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10]
    },
    "Decision Tree": {
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10]
    },
    # ... (see model_trainer.py for full config)
}
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'catboost'"
Install missing packages:
```bash
pip install catboost xgboost
```

### "File not found: notebook/data/StudentsPerformance.csv"
Ensure the dataset is placed at the correct path. Update the path in `data_ingestion.py` if needed.

### Slow Training
GridSearchCV with cv=5 is computationally intensive. To speed up:
- Reduce cv to 3 in `src/utils.py`
- Reduce hyperparameter grid size in `src/components/model_trainer.py`
- Use fewer models or disable hyperparameter tuning

---

## Project Status

✅ **Completed**:
- Data ingestion and train-test split
- Feature preprocessing and scaling
- 8 regression models with hyperparameter tuning
- Model evaluation and selection
- Artifact persistence

🔄 **In Development**:
- Model performance optimization
- Cross-validation analysis
- Feature importance analysis

📋 **Future Enhancements**:
- API endpoint for predictions
- Model explainability (SHAP values)
- Automated model retraining pipeline
- Production deployment

---

## Authors & Contribution

**Project**: Student Performance Indicator  
**Repository**: [GitHub - Student-Performance-Indicator](https://github.com/Rojeh-wael/Student-Performance-Indicator)

---

## License

See LICENSE file in the project root.

---

## Contact & Support

For issues, questions, or contributions, please open an issue on the GitHub repository.