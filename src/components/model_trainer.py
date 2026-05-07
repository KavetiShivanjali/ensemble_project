from operator import le
import os
import sys 
import argparse
from dataclasses import dataclass


from sklearn.preprocessing import LabelEncoder
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
import mlflow
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
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
            logging.info(f"Initiating training for: {model_type}")
            mlflow.set_experiment(experiment_name)

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_val, y_val = val_array[:, :-1], val_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_val = le.transform(y_val)
            y_test = le.transform(y_test)

            def objective(trial):
                with mlflow.start_run(run_name=f"{model_type}_trial_{trial.number}", nested=True):
                    
                    # 1. Capture Hyperparameters based on model_type argument
                    if model_type == 'logistic_regression':
                        params = {"C": trial.suggest_float("C", 1e-3, 5.0, log=True),
                                "solver": "lbfgs",  # Changed from 'liblinear' to 'lbfgs'
                                "class_weight": "balanced", # Explicitly handle multiple classes
                                "max_iter": 1000}
                        model = LogisticRegression(**params)

                    elif model_type == 'random_forest':
                        params = {
                            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                            "max_depth": trial.suggest_int("max_depth", 5, 20),
                            "class_weight": "balanced"
                        }
                        model = RandomForestClassifier(**params)

                    elif model_type == 'xgboost':
                        
                        params = {
                            "max_depth": trial.suggest_int("max_depth", 3, 10),
                            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 5)
                        }
                        model = XGBClassifier(**params)

                    elif model_type == 'catboost':
                        params = {
                            "depth": trial.suggest_int("depth", 4, 10),
                            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 4.0), # Fixed: Standard regularization value
                            "auto_class_weights": "Balanced" , # Let CatBoost handle class imbalance automatically
                            "verbose": False
                        }
                        model = CatBoostClassifier(**params)

                    elif model_type == 'decision_tree':
                        params = {"max_depth": trial.suggest_int("max_depth", 3, 15), 
                                  "class_weight": "balanced"}
                        model = DecisionTreeClassifier(**params)

                    # 2. Train and Calculate Metrics
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    
                    acc = accuracy_score(y_val, y_pred)
                    # We target recall for class 0 ('<30') specifically as requested
                    rec_lt30 = recall_score(y_val, y_pred, labels=[0], average='macro')

                    mlflow.log_params(params)
                    mlflow.log_metrics({"accuracy": acc, "recall_lt30": rec_lt30})

                    # 3. Objective: Prioritize Recall (0.7 weight) over Accuracy (0.3 weight)
                    return 0.7 * rec_lt30 + 0.3 * acc

            # Run Optuna Study
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)

            # Logging best results for the artifacts
            logging.info(f"Best params for {model_type}: {study.best_params}")

            # 1. Identify the Best Trial
            logging.info(f"Best Trial Score: {study.best_value}")
            best_params = study.best_params
            
            # Remove the 'classifier' key from params so it can be passed to the model constructor
            model_name = model_type
            
            # 2. Re-initialize the Best Model Type
            if model_name == 'logistic_regression':
                best_model = LogisticRegression(**best_params)
            elif model_name == 'random_forest':
                best_model = RandomForestClassifier(**best_params)
            elif model_name == 'xgboost':
                best_model = XGBClassifier(**best_params)
            elif model_name == 'catboost':
                best_model = CatBoostClassifier(**best_params,  verbose=False)
            elif model_name == 'decision_tree':
                best_model = DecisionTreeClassifier(**best_params)

            # 3. Final Fit on the full training array
            logging.info(f"Re-training the best model: {model_name}")
            best_model.fit(X_train, y_train)

            y_pred = best_model.predict(X_test)
            acc_score = accuracy_score(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)

            logging.info(f"Final evaluation on test set for best model ({model_name}):")
            logging.info(f"Accuracy Score: {acc_score}")
            logging.info(f"Classification Report:\n{class_report}")
            logging.info(f"Confusion Matrix:\n{conf_matrix}")

            # 4. Save as Pickle in Artifacts
            # This uses the path defined in your DataTransformationConfig (e.g., 'artifacts/model.pkl')
            save_object(
                file_path=self.model_trainer_config.get_model_path(model_name),
                obj=best_model
            )
            
            logging.info(f"Best model ({model_name}) saved successfully at {self.model_trainer_config.get_model_path(model_name)}")
            return self.model_trainer_config.get_model_path(model_name)

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
