#categorical to numerical features
import sys 
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
import category_encoders as ce
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object
import warnings
warnings.filterwarnings("ignore")

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def apply_scaling(self, X_train, X_val, X_test):
        """
        Fits scalers on training data and transforms all sets.
        Prevents data leakage for the Ensemble assessment.
        """
        # 1. Define feature groups based on prior Box Plot analysis
        # Skewed features with extreme outliers
        robust_features = ['number_outpatient', 'number_emergency', 'number_inpatient', 'score']
        
        # Well-behaved or log-normalized features
        standard_features = [
            'num_lab_procedures', 'num_medications', 
            'time_in_hospital', 'num_procedures', 'number_diagnoses'
        ]

        # 2. Define the ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('robust', RobustScaler(), robust_features),
                ('standard', StandardScaler(), standard_features)
            ],
            remainder='passthrough'  # Keep categorical columns for encoding later
        )

        # 3. Fit on TRAIN only to prevent leakage
        preprocessor.fit(X_train)

        # 4. Transform all sets
        # Note: ColumnTransformer returns a numpy array; we convert back to DataFrame for readability
        cols = robust_features + standard_features + [c for c in X_train.columns if c not in robust_features + standard_features]
        
        X_train_scaled = pd.DataFrame(preprocessor.transform(X_train), columns=cols, index=X_train.index)
        X_val_scaled = pd.DataFrame(preprocessor.transform(X_val), columns=cols, index=X_val.index)
        X_test_scaled = pd.DataFrame(preprocessor.transform(X_test), columns=cols, index=X_test.index)

        return X_train_scaled, X_val_scaled, X_test_scaled, preprocessor
    
    def apply_categorical_encoding(self,X_train, X_val, X_test, y_train):
        """
        Applies a hybrid encoding strategy to handle high-cardinality clinical features.
        Demonstrates 'Applied Science' rigor by preventing data leakage.
        """
        # 1. Identify High-Cardinality Features (>10 unique values)
        high_card_features = ['diag_1', 'diag_2', 'diag_3', 'medical_specialty']
        
        # 2. Identify Low-Cardinality Features
        numerical_cols = ["time_in_hospital","num_lab_procedures","num_procedures","num_medications","number_outpatient","number_emergency","number_inpatient","number_diagnoses"]
        low_card_features = list(set(X_train.columns)-set(numerical_cols)-set(high_card_features))

        # 3. Target Encoding with Smoothing for High-Cardinality
        # Fitting ONLY on X_train and y_train to prevent leakage
        target_enc = ce.TargetEncoder(cols=high_card_features, smoothing=10)
        target_enc.fit(X_train[high_card_features], y_train)

        # 4. Ordinal/Label Encoding for features with inherent order (like Age)
        # Note: For non-ordinal low-card, One-Hot Encoding is often preferred
        label_enc = ce.OrdinalEncoder(cols=low_card_features)
        label_enc.fit(X_train[low_card_features])

        # 5. Transform all sets
        def transform_sets(X):
            X_encoded = X.copy()
            X_encoded[high_card_features] = target_enc.transform(X[high_card_features])
            X_encoded[low_card_features] = label_enc.transform(X[low_card_features])
            return X_encoded

        return transform_sets(X_train), transform_sets(X_val), transform_sets(X_test)


    def get_data_transformer_object(self, df):
        try:
            logging.info("Data Transformation initiated")

            outlier_num_features = ['number_outpatient', 'number_emergency', 'number_inpatient']
        
            # Well-behaved or log-normalized features
            normal_num_features = [
                'num_lab_procedures', 'num_medications', 
                'time_in_hospital', 'num_procedures', 'number_diagnoses'
            ]

            high_card_features = ['diag_1', 'diag_2', 'diag_3', 'medical_specialty']
            low_card_features = list(set(df.columns)-set(outlier_num_features + normal_num_features))

            target_column_name = 'readmitted'
            
            outlier_num_pipeline = Pipeline(steps=[
                ('scaler', RobustScaler())
            ])

            normal_num_pipeline = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])

            high_card_features_pipeline = Pipeline(steps=[
                ('target_enc', ce.TargetEncoder(cols=high_card_features, smoothing=10))
            ])

            low_card_features_pipeline = Pipeline(steps=[
                ('label_enc', ce.OrdinalEncoder(cols=low_card_features))
            ])


            
            logging.info(f"High cardinality categorical columns: {high_card_features}")
            logging.info(f"Low cardinality categorical columns: {low_card_features}")
            logging.info(f"Non outlier numerical columns: {normal_num_features}")
            logging.info(f"Outlier numerical columns: {outlier_num_features}")
            preprocessor = ColumnTransformer([
                ('outlier_num_pipeline', outlier_num_pipeline, outlier_num_features),
                ('normal_num_pipeline', normal_num_pipeline, normal_num_features),
                ('high_card_features_pipeline', high_card_features_pipeline, high_card_features),
                ('low_card_features_pipeline', low_card_features_pipeline, low_card_features)
            ])
            return preprocessor
        
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, val_path, test_path):  
        try:
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train val and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object(train_df.drop(columns=['readmitted'], axis=1))

            target_column_name = 'readmitted'

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_val_df = val_df.drop(columns=[target_column_name], axis=1)
            target_feature_val_df = val_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe validation dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df, target_feature_train_df)
            input_feature_val_arr = preprocessing_obj.transform(input_feature_val_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            val_arr = np.c_[input_feature_val_arr, np.array(target_feature_val_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                val_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)   