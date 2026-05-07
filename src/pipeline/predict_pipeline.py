import os
import sys
import argparse
import pandas as pd
import numpy as np
import lime
import lime.lime_tabular
from datetime import datetime
from src.utils import load_object
from src.exception import CustomException
from src.logger import logging
import warnings
warnings.filterwarnings("ignore")

class PredictPipeline:
    def __init__(self):
        pass

    def preprocess_raw_data(self, df):
        """
        Drops columns not used during training to maintain feature consistency.
        """
        try:
            # Columns dropped based on data cleaning requirements
            cols_to_drop = ['encounter_id', 'patient_nbr', 'weight', 'max_glu_serum']
            existing_cols = [col for col in cols_to_drop if col in df.columns]
            df_cleaned = df.drop(columns=existing_cols, axis=1)
            return df_cleaned
        except Exception as e:
            raise CustomException(e, sys)

    def run_batch_prediction_and_explain(self, test_file, model_type, row_to_explain):
        try:
            logging.info(f"Initiating batch prediction for {model_type}")

            # 1. Generate Timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 2. Load Data and Artifacts
            data_path = os.path.join("data", test_file)
            df = pd.read_csv(data_path)
            
            model_path = os.path.join("artifacts\model", f"{model_type}_model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # 3. Preprocess and Predict Entire Dataset
            X_raw = self.preprocess_raw_data(df.drop(columns=['readmitted'], axis=1, errors='ignore'))
            X_test_transformed = preprocessor.transform(X_raw)
            
            logging.info("Running predictions on the entire test set...")
            predictions = model.predict(X_test_transformed)
            
            # Map numeric predictions back to labels
            class_map = {0: '<30', 1: '>30', 2: 'NO'}
            # Robust extraction of prediction values from array
            df['prediction_label'] = [class_map[int(p[0]) if isinstance(p, (np.ndarray, list)) else int(p)] for p in predictions]

            # 4. Store Results with Timestamp in artifacts\predict
            output_dir = os.path.join("artifacts", "predict")
            os.makedirs(output_dir, exist_ok=True)
            
            predictions_filename = f"{model_type}_results_{timestamp}.csv"
            predictions_file = os.path.join(output_dir, predictions_filename)
            df.to_csv(predictions_file, index=False)
            logging.info(f"Full predictions saved to {predictions_file}")

            # 5. Explain a Specific Row via LIME
            logging.info(f"Generating LIME explanation for row index: {row_to_explain}")
            
            feature_names = preprocessor.get_feature_names_out()
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_test_transformed,
                feature_names=feature_names,
                class_names=['<30', '>30', 'NO'],
                mode='classification'
            )

            # Safety check for index bounds
            row_idx = min(row_to_explain, len(X_test_transformed) - 1)
            exp = explainer.explain_instance(X_test_transformed[row_idx], model.predict_proba)

            # Save LIME HTML with Timestamp
            lime_filename = f"{model_type}_row_{row_idx}_explanation_{timestamp}.html"
            lime_path = os.path.join(output_dir, lime_filename)
            exp.save_to_file(lime_path)
            
            print(f"\nBatch prediction completed.")
            print(f"Results CSV: {predictions_filename}")
            print(f"LIME Report: {lime_filename}\n")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., catboost)')
    parser.add_argument('--file', type=str, required=True, help='CSV file in data folder')
    parser.add_argument('--row', type=int, default=0, help='Row index to explain')
    args = parser.parse_args()

    pipeline = PredictPipeline()
    pipeline.run_batch_prediction_and_explain(args.file, args.model, args.row)
    