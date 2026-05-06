import os
import sys
import argparse
import pandas as pd
import numpy as np
import lime
import lime.lime_tabular
from sklearn.metrics import accuracy_score, classification_report, recall_score
from src.utils import load_object
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import LabelEncoder

class ModelEvaluator:
    def __init__(self):
        pass

    def preprocess_data(self, df, target_column):
        """
        Handles all data cleaning: dropping IDs, weight, high-null columns, and rows with nulls.
        """
        try:
            logging.info("Starting data preprocessing steps...")
            
            # Dropping columns based on project requirements: IDs, Weight, and max_glu_serum
            cols_to_drop = ['encounter_id', 'patient_nbr', 'weight', 'max_glu_serum']
            existing_cols = [col for col in cols_to_drop if col in df.columns]
            
            df = df.drop(columns=existing_cols, axis=1)
            df = df.dropna()
            
            logging.info(f"Preprocessing complete. Dropped: {existing_cols}. Rows remaining: {len(df)}")

            # Separate features and target
            X = df.drop(columns=[target_column], axis=1)
            y = df[target_column]
            
            return X, y

        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_and_explain(self, raw_test_df, model_type, target_column, row_index):
        try:
            logging.info(f"--- Evaluating {model_type} for Row {row_index} ---")
            
            # 1. Centralized Preprocessing
            X_test_raw, y_test = self.preprocess_data(raw_test_df, target_column)

            # Encode labels to match the numeric format expected by the model
            le = LabelEncoder()
            y_test = le.fit_transform(y_test)

            # 2. Load Artifacts
            model_path = os.path.join("artifacts", f"{model_type}_model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # 3. Transform Test Data
            X_test_transformed = preprocessor.transform(X_test_raw)
            
            # 4. Predictions & Metrics
            y_pred = model.predict(X_test_transformed)
            acc = accuracy_score(y_test, y_pred)
            
            # Focused on Recall for the <30 readmission class (mapped to 0)
            rec_lt30 = recall_score(y_test, y_pred, labels=[0], average='macro')
            
            # Print Global Stats to Console for immediate verification
            print("\n" + "="*20 + " CLASSIFICATION REPORT " + "="*20)
            print(f"Accuracy: {acc:.4f} | Recall (<30): {rec_lt30:.4f}")
            print("-" * 63)
            print(classification_report(y_test, y_pred))

            # 5. LIME Explainability
            feature_names = preprocessor.get_feature_names_out()
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_test_transformed, 
                feature_names=feature_names,
                class_names=['<30', '>30', 'NO'],
                mode='classification'
            )

            # Row safety check
            row_index = min(row_index, len(X_test_transformed) - 1)
            exp = explainer.explain_instance(X_test_transformed[row_index], model.predict_proba)

            # Print local LIME weights to Console
            print("="*20 + f" LIME EXPLANATION (ROW {row_index}) " + "="*19)
            for feature, weight in exp.as_list():
                print(f"{feature:<40} | {weight:>15.4f}")
            print("="*63 + "\n")

            # 6. Save LIME Result strictly as HTML
            html_path = os.path.join("artifacts", f"{model_type}_row_{row_index}_lime.html")
            exp.save_to_file(html_path)
            logging.info(f"Interactive LIME report saved to: {html_path}")

            return acc, rec_lt30

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., catboost)')
    parser.add_argument('--test_file', type=str, required=True, help='Name of test CSV in data folder')
    parser.add_argument('--row', type=int, default=0, help='Row index to explain')
    args = parser.parse_args()

    try:
        test_data_path = os.path.join("data", args.test_file)
        
        if os.path.exists(test_data_path):
            test_df = pd.read_csv(test_data_path)
            evaluator = ModelEvaluator()
            evaluator.evaluate_and_explain(test_df, args.model, 'readmitted', args.row)
        else:
            logging.error(f"Test data not found at: {test_data_path}")

    except Exception as e:
        raise CustomException(e, sys)