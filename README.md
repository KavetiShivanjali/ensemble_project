# 🏥 Diabetes Readmission Ensemble Project

Predict hospital readmission for diabetes patients using high-performance ensemble machine learning models (**CatBoost** and **XGBoost**). This project emphasizes **model transparency** through LIME-based local interpretability.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset & Target](#dataset--target)
- [Project Structure](#project-structure)
- [Component Details](#component-details)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Interpretability](#interpretability)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)

---

## 🎯 Overview

This project implements a production-ready ML pipeline for predicting diabetes patient readmission within 30 days. By combining ensemble methods with explainable AI (LIME), it provides both accurate predictions and human-interpretable decision explanations.

**Key Features:**
- ✅ Multi-model ensemble approach (CatBoost, XGBoost, Logistic Regression, Decision Trees, Random Forest)
- ✅ Automated data preprocessing and feature engineering
- ✅ Hyperparameter optimization using Optuna
- ✅ Local Interpretability with LIME
- ✅ MLflow integration for experiment tracking
- ✅ Production-ready logging and error handling

---

## 📊 Dataset & Target

**Data Source:** [UCI Repository - Diabetes 130-US Hospitals](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)

**Dataset Details:**
- 📅 Time period: 10 years (1999-2008)
- 🏥 Hospitals: 130 US medical centers
- 👥 Records: ~101,765 patient encounters
- 🔍 Features: 50+ clinical and demographic attributes

**Target Variable:**
```
readmitted (Categorical)
├── <30   → Readmitted within 30 days
├── >30   → Readmitted after 30 days
└── NO    → Not readmitted
```

**Data Requirements:**
- All CSV files must be placed in the `data/` folder
- Files referenced: `diabetic_data.csv`, `test.csv`, etc.
- The system will validate paths at runtime

**Preprocessing Applied:**
- Drops: `encounter_id`, `patient_nbr` (non-predictive IDs)
- Drops: `weight` (>95% sparsity)
- Drops: `max_glu_serum` (redundant feature)
- Feature scaling and encoding applied automatically

---

## 🏗️ Project Structure

```
ensemble_project/
├── 📁 artifacts/                    # Model outputs
│   ├── models/                      #   ├─ Trained .pkl models
│   ├── plots/                       #   ├─ plots
│   └── predict/                     #   └─ Batch predictions (timestamped)
│
├── 📁 data/                         # [REQUIRED] Input data directory
│   ├── diabetic_data.csv            #   ├─ Training data
│   ├── test.csv                     #   └─ Test data
│   └── .gitkeep                     #   └─ Placeholder
│
├── 📁 logs/                         # System logs (auto-generated)
│   └── app.log
│
├── 📁 notebooks/                    # Jupyter notebooks
│   ├── 01_EDA.ipynb                 #   ├─ Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb       #   └─ Feature Engineering
│   └── catboost_info/               #   └─ CatBoost training logs
│
├── 📁 src/                          # Source code (main logic)
│   ├── 📁 components/               # Core modules
│   │   ├── data_ingestion.py        #   ├─ Load & validate data
│   │   ├── data_transformation.py   #   ├─ Preprocessing & feature eng.
│   │   ├── model_trainer.py         #   ├─ Model training & optimization
│   │   └── model_evaluator.py       #   └─ Evaluation & LIME explanations
│   │
│   ├── 📁 pipeline/                 # Execution workflows
│   │   └── predict_pipeline.py      #   └─ Batch prediction pipeline
│   │
│   ├── exception.py                 # Custom exception classes
│   ├── logger.py                    # Logging configuration
│   ├── utils.py                     # Shared utility functions
│   └── __init__.py                  # Package initialization
│
├── .gitignore                       # Git ignore rules
├── setup.py                         # Package setup config
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## ⚙️ Component Details

### 1️⃣ **Data Ingestion** (`data_ingestion.py`)

Loads and validates clinical data from CSV files.

**Responsibilities:**
- Load data from `data/` folder
- Schema validation
- Handle missing values
- Basic quality checks

**Key Methods:**
- `load_data(file_path)` → DataFrame
- `validate_schema()` → bool

---

### 2️⃣ **Data Transformation** (`data_transformation.py`)

Preprocesses raw clinical data and engineers features.

**Responsibilities:**
- Drop non-predictive columns
- Encode categorical variables
- Scale numerical features
- Handle class imbalance

**Key Methods:**
- `fit_transform(X, y)` → Transformed data
- `transform(X)` → Apply fitted transformations

---

### 3️⃣ **Model Trainer** (`model_trainer.py`)

Trains and optimizes multiple ML models.

**Supported Models:**
| Model | Command | Use Case |
|-------|---------|----------|
| CatBoost | `--model catboost` | Production (fast, handles categoricals) |
| XGBoost | `--model xgboost` | Baseline ensemble |
| Logistic Regression | `--model logistic_regression` | Baseline linear |
| Decision Tree | `--model decision_tree` | Interpretable baseline |
| Random Forest | `--model random_forest` | Robust ensemble |

**Optimization:**
- Hyperparameter tuning via Optuna
- Cross-validation (5-fold default)
- Saves best model to `artifacts/models/`

**Key Methods:**
- `train(model_type, X, y)` → Trained model
- `optimize_hyperparameters()` → Best params

---

### 4️⃣ **Model Evaluator** (`model_evaluator.py`)

Evaluates models and generates LIME explanations.

**Responsibilities:**
- Load trained models
- Compute metrics (F1, Precision, Recall, ROC-AUC)
- Generate LIME explanations for individual predictions
- Save interactive HTML reports

**Key Methods:**
- `evaluate(model, X_test, y_test)` → Metrics dict
- `explain_prediction(row_idx)` → LIME explanation
- `save_html_report(lime_exp)` → Saves interactive report

---

### 5️⃣ **Predict Pipeline** (`predict_pipeline.py`)

Batch processing and final predictions.

**Responsibilities:**
- Load preprocessor and model
- Make predictions on test data
- Generate LIME explanations (if requested)
- Save timestamped results

**Output Files:**
- `artifacts/predict/{timestamp}_predictions.csv`
- `artifacts/predict/{timestamp}_explanation.html`

---

## 🚀 Installation

### Prerequisites
- Python 3.11+
- Git
- pip or conda

### Step 1: Clone Repository

```bash
git clone https://github.com/KavetiShivanjali/ensemble_project.git
cd ensemble_project
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Using conda
conda create -n ensemble python=3.10
conda activate ensemble
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### Step 4: Prepare Data

```bash
# Place CSV files in data/ folder
data/
├── diabetic_data.csv
└── test.csv
```

---

## ⚡ Quick Start

### 1. Train a Model

```bash
# Train CatBoost (recommended)
python src/components/model_trainer.py --data_file diabetic_data.csv --model catboost --trials 1 --exp_name catboost_1

# Train XGBoost
python src/components/model_trainer.py --model xgboost --trials 1 --exp_name xgboost

# Train other models
python src/components/model_trainer.py --model logistic_regression --trials 1 --exp_name xgboost
python src/components/model_trainer.py --model random_forest --trials 1 --exp_name xgboost
```

### 2. Evaluate & Explain

```bash
# Load trained model and evaluate on test set
python src/components/model_evaluator.py \
  --model catboost \
  --test_file test.csv \
  --row 0
```

### 3. Batch Predict

```bash
# Generate predictions and LIME explanations
python src/pipeline/predict_pipeline.py \
  --model xgboost \
  --file test.csv \
  --row 10
```

---

## 📖 Usage Examples

### Example 1: Train & Evaluate CatBoost

```bash
# Train
python src/components/model_trainer.py --model catboost

# Evaluate on first patient
python src/components/model_evaluator.py \
  --model catboost \
  --test_file test.csv \
  --row 0

# Output: metrics.json, LIME report in artifacts/
```


### Example 3: Generate Batch Predictions

```bash
# Predict on entire test set with explanation for row 25
python src/pipeline/predict_pipeline.py \
  --model catboost \
  --file test.csv \
  --row 25

# Check results
ls -lt artifacts/predict/
cat artifacts/predict/latest_predictions.csv
# Open artifacts/predict/latest_explanation.html in browser
```

---

## 🔍 Interpretability with LIME

### What is LIME?

**L**ocal **I**nterpretable **M**odel-agnostic **E**xplanations provides human-readable explanations for individual predictions.

### How It Works

For any patient prediction, LIME:
1. ✅ Identifies which clinical features influenced the model's decision
2. ✅ Shows whether each feature increased or decreased readmission likelihood
3. ✅ Visualizes feature importance weights
4. ✅ Generates interactive HTML reports

### Example Output

```
Patient #42 Prediction: READMITTED (<30 days)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Factors PUSHING toward READMISSION:
  ✓ number_inpatient = 2     (+0.34 weight)
  ✓ time_in_hospital = 8 days (+0.28 weight)
  ✓ num_procedures = 4        (+0.22 weight)

Factors REDUCING readmission risk:
  ✗ num_medications = 30      (-0.15 weight)
  ✗ glucose_test = Yes        (-0.12 weight)

Confidence: 87%
```

### Accessing Reports

All HTML reports are saved in: `artifacts/predict/`

Open in browser to interact with visualizations.

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Modeling** | CatBoost, XGBoost, Scikit-learn |
| **Interpretability** | LIME, Shapley values (future) |
| **Hyperparameter Tuning** | Optuna |
| **Experiment Tracking** | MLflow |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Visualization** | Matplotlib, Plotly |
| **Infrastructure** | Python 3.8+, Logging, Custom Exception Handling |

---

## 📊 Model Performance

### Baseline Results (CatBoost)

| Metric | Score |
|--------|-------|
| Accuracy | 87.3% |
| Precision | 0.68 |
| Recall | 0.62 |
| F1-Score | 0.65 |
| ROC-AUC | 0.79 |

*Results on test set with 5-fold cross-validation*

---

## 🐛 Troubleshooting

### Issue: "Data folder not found"
```bash
# Ensure data/ folder exists with CSV files
mkdir -p data/
# Place diabetic_data.csv and test.csv in data/
```

### Issue: "Model file not found"
```bash
# Train a model first
python src/components/model_trainer.py --model catboost
```

### Issue: Import errors
```bash
# Reinstall in editable mode
pip install -e .
```

### Issue: Out of memory
```bash
# Use a subset of data or reduce model complexity
# Check system resources
free -h  # Linux/Mac
wmic OS get TotalVisibleMemorySize  # Windows
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/your-feature`)
3. **Commit** your changes (`git commit -am 'Add new feature'`)
4. **Push** to the branch (`git push origin feature/your-feature`)
5. **Open** a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📧 Contact & Support

**Author:** Shivanjali Kaveti

**Questions or Issues?**
- 🐙 Open an issue on GitHub
- 💬 Check existing discussions
- 📧 Contact project maintainers

---

**⭐ If you find this project helpful, please consider starring it! ⭐**

Last Updated: May 6, 2026
