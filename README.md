# 🎓 Student Performance Predictor

## 📌 Overview
This project predicts student scores (math, reading, writing) using machine learning.  
It includes a complete ML pipeline:
- Data loading & preprocessing
- Feature engineering
- Model training & tuning
- Model evaluation

---

## ⚙️ What It Does
- Processes student data (categorical → encoded & scaled)
- Trains 8 regression models
- Uses GridSearchCV to find the best model
- Evaluates using R² score
- Saves:
  - `model.pkl` → best model
  - `preprocessor.pkl` → preprocessing pipeline

---

## 📂 Project Structure
```
├── artifacts/        # Outputs (data, model, preprocessor)
├── notebook/         # EDA & experiments
├── src/              # Main code
│   ├── components/   # Pipeline steps
│   ├── utils.py      # Helpers
│   ├── logger.py
│   └── exception.py
├── requirements.txt
└── README.md
```

---

## 📊 Dataset
- **1000** student records
- **Features:** gender, race, education, lunch, test prep
- **Targets:** math, reading, writing scores

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add dataset
Place file here:
```
notebook/data/StudentsPerformance.csv
```

### 3. Run pipeline
```bash
python -m src.components.data_ingestion
```

---

## 🔁 Pipeline Steps

| Step | Description |
|------|-------------|
| **1. Data Ingestion** | Load & split data (80/20) |
| **2. Data Transformation** | Encode categorical features & scale |
| **3. Model Training** | Train 8 models, tune with GridSearchCV, select best |

---

## 🧠 Models Used
- Random Forest
- Decision Tree
- Gradient Boosting
- Linear Regression
- XGBoost
- CatBoost
- AdaBoost
- KNN

---

## 📈 Output
- Best model saved → `artifacts/model.pkl`
- Preprocessor saved → `artifacts/preprocessor.pkl`
- Metric → **R² score**

---

## ⚠️ Notes
- Training may take **5–10 minutes**
- If slow:
  - Reduce CV folds
  - Reduce parameter grid

---

## 🔮 Future Work
- [ ] API for predictions
- [ ] Model explainability (SHAP)
- [ ] Deployment

---

## 👤 Author
**Student Performance Indicator Project**  
GitHub: [https://github.com/Rojeh-wael/Student-Performance-Indicator](https://github.com/Rojeh-wael/Student-Performance-Indicator)
