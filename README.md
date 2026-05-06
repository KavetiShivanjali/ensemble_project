## Diabetes Readmission Ensemble Project
This project predicts hospital readmission for diabetes patients using high-performance ensemble models (CatBoost and XGBoost). It emphasizes model transparency through LIME (Local Interpretable Model-agnostic Explanations) and follows a modular, production-ready architecture.

📊 Dataset & Target
The project utilizes clinical data representing 10 years of care at 130 US hospitals.

UCI Dataset Link: Diabetes 130-US hospitals (UCI Repository)

Target Variable: readmitted (Categorical: <30, >30, NO).

Data Requirements: All CSV files (e.g., diabetic_data.csv, test.csv) MUST be stored in the data/ folder within the project root for the pipelines to function correctly.

Preprocessing: The system automatically drops non-predictive IDs (encounter_id, patient_nbr), weight (high sparsity), and max_glu_serum to maintain model integrity.

🏗️ Project Structure
Based on the modular architecture , the project is organized as follows:

Plaintext
ENSEMBLE_PROJECT/
├── artifacts/               # Saved models (.pkl), preprocessors, and LIME reports
├── data/                    # REQUIRED: All CSV files must be stored here
├── logs/                    # Automated system logs
├── notebooks/               # EDA and experimentation
├── src/                     # Source code directory
│   ├── components/          # Core functional modules
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluator.py
│   ├── pipeline/            # Execution workflows
│   │   └── predict_pipeline.py
│   ├── exception.py         # Custom error handling
│   ├── logger.py            # Logging configuration
│   └── utils.py             # Shared utility functions
├── setup.py                 # Package configuration
└── requirements.txt         # Project dependencies
⚙️ Component & Parameter Details
1. Model Trainer (model_trainer.py)
Responsible for training models and optimizing performance.

--model: Specify algorithm.

catboost: Trains using the CatBoost classifier.

xgboost: Trains using the XGBoost classifier.

logistic_regression: Trains using Logistic regression

decision_tree: Trains using Decision tree.

random_forest: Trains using Random forest.

Logic: Performs hyperparameter tuning and saves the final model to artifacts/.

2. Model Evaluator (model_evaluator.py)
Assesses performance and provides local interpretability.

--model: Select catboost or xgboost model to load from artifacts.

--test_file: CSV filename (e.g., test.csv) located in the data/ folder.

--row: Integer index of a specific patient for LIME explanation.

3. Predict Pipeline (predict_pipeline.py)
Designed for batch processing and generating final outputs.

--model: Model type to use (catboost or xgboost).

--file: CSV filename in the data/ folder.

--row: Specific index to explain via LIME.

Output: Saves timestamped results to artifacts/predict/ and generates an interactive HTML explanation.

🚀 Installation & Execution
1. Setup from GitHub
Bash
git clone https://github.com/your-username/ENSEMBLE_PROJECT.git
cd ENSEMBLE_PROJECT

# Install dependencies and project in editable mode
pip install -r requirements.txt
pip install -e .
2. Run the Pipelines
CMD
# Train a model
python src/components/model_trainer.py --model catboost

# Evaluate and explain row 0
python src/components/model_evaluator.py --model catboost --test_file test.csv --row 0

# Batch predict and explain row 10
python src/pipeline/predict_pipeline.py --model xgboost --file test.csv --row 10
🔍 Interpretability (LIME)
For any individual prediction, the system generates an interactive HTML report. This displays which clinical features (e.g., number_inpatient) pushed the model toward a specific classification, translating "black-box" decisions into interpretable weights.

🛠️ Tech Stack
Modeling: CatBoost, XGBoost, Scikit-learn.

Interpretability: LIME.

Management: MLflow, Optuna.

Infrastructure: Python 3.x, Logging, Custom Exception Handling.
