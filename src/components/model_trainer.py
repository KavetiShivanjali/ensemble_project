from operator import le
import os
import sys 
import argparse
from dataclasses import dataclass

import numpy as np
from fastapi import params
from sklearn.preprocessing import LabelEncoder
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix, recall_score
import mlflow
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")


@dataclass
class ModelTrainerConfig:
    def get_model_path(self, model_name):
        output_dir = os.path.join("artifacts", "model")
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join('artifacts\model', f"{model_name}_model.pkl")
    trained_model_file_path: str = ''

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, val_array, test_array, model_type='catboost', n_trials=20, experiment_name='Model_Search'):
            try:
                logging.info(f"Initiating training for: {model_type} with 5-Fold Cross Validation")
                mlflow.set_experiment(experiment_name)

                # Separate Features and Target
                X_train, y_train = train_array[:, :-1], train_array[:, -1]
                X_test, y_test = test_array[:, :-1], test_array[:, -1]
                
                # Label Encoding
                le = LabelEncoder()
                y_train = le.fit_transform(y_train)
                y_test = le.transform(y_test)

                def objective(trial):
                    # Move StratifiedKFold inside the objective to get CV-based scores
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    cv_recalls = []
                    cv_accuracies = []

                    # Define Parameters
                    if model_type == 'logistic_regression':
                        params = {"C": trial.suggest_float("C", 1e-3, 5.0, log=True), "solver": "lbfgs", "class_weight": "balanced", "max_iter": 1000}
                    elif model_type == 'random_forest':
                        params = {"n_estimators": trial.suggest_int("n_estimators", 50, 200), "max_depth": trial.suggest_int("max_depth", 5, 20), "class_weight": "balanced"}
                    elif model_type == 'xgboost':
                        params = {"max_depth": trial.suggest_int("max_depth", 3, 10), "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1), "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 5)}
                    elif model_type == 'catboost':
                        params = {"depth": trial.suggest_int("depth", 4, 10), "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1), "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 4.0), "auto_class_weights": "Balanced", "verbose": False}
                    elif model_type == 'decision_tree':
                        params = {"max_depth": trial.suggest_int("max_depth", 3, 15), "class_weight": "balanced"}

                    # Start MLflow nested run for the trial
                    with mlflow.start_run(run_name=f"{model_type}_trial_{trial.number}", nested=True):
                        # Cross-Validation Loop
                        for train_idx, val_idx in skf.split(X_train, y_train):
                            # Convert to DataFrame if they are currently numpy arrays for slicing
                            X_t, X_v = X_train[train_idx], X_train[val_idx]
                            y_t, y_v = y_train[train_idx], y_train[val_idx]

                            # Initialize and fit
                            if model_type == 'catboost': model = CatBoostClassifier(**params)
                            elif model_type == 'xgboost': model = XGBClassifier(**params)
                            elif model_type == 'logistic_regression': model = LogisticRegression(**params)
                            elif model_type == 'random_forest': model = RandomForestClassifier(**params)
                            else: model = DecisionTreeClassifier(**params)

                            model.fit(X_t, y_t)
                            y_pred = model.predict(X_v)

                            cv_accuracies.append(accuracy_score(y_v, y_pred))
                            cv_recalls.append(recall_score(y_v, y_pred, labels=[0], average='macro'))

                        # Calculate Mean CV Metrics
                        mean_acc = np.mean(cv_accuracies)
                        mean_rec = np.mean(cv_recalls)

                        mlflow.log_params(params)
                        mlflow.log_metrics({"cv_accuracy": mean_acc, "cv_recall_lt30": mean_rec})

                        logging.info(f"Trial {trial.number} - CV Accuracy: {mean_acc:.4f}, CV Recall (<30 days): {mean_rec:.4f}")
                        logging.info(f"Trial {trial.number} - Parameters: {params}")
                        logging.info(f"Trial {trial.number} - Classification Report:\n{classification_report(y_v, y_pred)}")
                        logging.info(f"Trial {trial.number} - Confusion Matrix:\n{confusion_matrix(y_v, y_pred)}")
                        logging.info(f"Trial {trial.number} - Weighted Objective: {0.7 * mean_rec + 0.3 * mean_acc:.4f}")

                        # Weighted Objective: 70% Recall, 30% Accuracy
                        return 0.7 * mean_rec + 0.3 * mean_acc

                # Run Optuna Study
                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=n_trials)

                logging.info(f"Best Trial Score: {study.best_value}")
                best_params = study.best_params
                
                # Re-initialize Best Model with winning parameters
                if model_type == 'logistic_regression': best_model = LogisticRegression(**best_params)
                elif model_type == 'random_forest': best_model = RandomForestClassifier(**best_params)
                elif model_type == 'xgboost': best_model = XGBClassifier(**best_params)
                elif model_type == 'catboost': best_model = CatBoostClassifier(**best_params, verbose=False)
                elif model_type == 'decision_tree': best_model = DecisionTreeClassifier(**best_params)

                # Final Fit on the full training set
                logging.info(f"Re-training the best {model_type} model on full training data")
                best_model.fit(X_train, y_train)

                # Final Evaluation on the unseen Test Set
                y_pred_test = best_model.predict(X_test)
                logging.info(f"Test Accuracy: {accuracy_score(y_test, y_pred_test)}")
                logging.info(f"Test Confusion Matrix:\n{confusion_matrix(y_test, y_pred_test)}")

                # Save Object
                save_object(
                    file_path=self.model_trainer_config.get_model_path(model_type),
                    obj=best_model
                )
                
                return self.model_trainer_config.get_model_path(model_type)

            except Exception as e:
                raise CustomException(e, sys)

   

if __name__ == "__main__":
    # 1. Initialize the ArgumentParser
    parser = argparse.ArgumentParser(description="Diabetes Readmission Pipeline")

    # 2. Add model-related arguments
    # You can specify the model type (e.g., CatBoost vs LogReg) and Optuna trials
    parser.add_argument('--data_file', type=str, default='diabetic_data.csv', help='Path to the data file')
    parser.add_argument('--model', type=str, default='catboost', help='Model type to train')
    parser.add_argument('--trials', type=int, default=20, help='Number of Optuna search trials')
    parser.add_argument('--exp_name', type=str, default='Default_Exp', help='MLflow experiment name')

    # 3. Parse the arguments from the command line
    args = parser.parse_args()

    # --- Pipeline Execution ---
    data_ingestion = DataIngestion()
    train_data, val_data, test_data = data_ingestion.initiate_data_ingestion(data_file=args.data_file)

    data_transformation = DataTransformation()
    train_arr, val_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, val_data, test_data
    )

    # 4. Pass the captured arguments into the Model Trainer
    model_trainer = ModelTrainer()
    print(f"Starting {args.model} training with {args.trials} trials...")
    
    model_trainer.initiate_model_trainer(
        train_array=train_arr, 
        val_array=val_arr,  # Ensure you pass val_arr for hyperparameter tuning
        test_array=test_arr,  # Pass test_arr for final evaluation
        model_type=args.model,
        n_trials=args.trials,
        experiment_name=args.exp_name
    )
