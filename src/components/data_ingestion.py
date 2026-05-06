# splitting the data into train and test
import os 
import sys

from mlflow import data
from plotly import data
from narwhals.selectors import datetime
from pyparsing import col 
from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import save_object
import math 
from dataclasses import dataclass
from scipy.stats.mstats import winsorize
from datetime import datetime  

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    val_data_path: str = os.path.join('artifacts', 'val.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    # Converting '?' to NaN for all columns in the dataset
    def convert_to_none(self,x):
        if x == '?':
            return np.nan
        else:
            return x

    def impute_mode_by_group(self, df, target_col, group_col):
        """
        Imputes missing values in target_col using the mode 
        within each group of group_col.
        """
        # Calculate the mode for each group
        # Note: .mode() returns a Series, so we take the first element [0]
        group_modes = df.groupby(group_col)[target_col].transform(
            lambda x: x.mode()[0] if not x.mode().empty else np.nan
        )
        
        # Fill NA values
        df[target_col] = df[target_col].fillna(group_modes)
        
        # Fallback: if a group is entirely NaN, fill with the global mode
        df[target_col] = df[target_col].fillna(df[target_col].mode()[0])
        
        return df   


    def find_best_partition_col(self,df, target_col, candidate_cols):
        """
        Identifies which candidate column has the strongest 
        statistical association with the target_col.
        """
        best_col = None
        best_chi2 = -1

        # Drop NAs temporarily for the statistical test
        temp_df = df[[target_col] + candidate_cols].dropna()

        for col in candidate_cols:
            # Create a contingency table
            contingency_table = pd.crosstab(temp_df[target_col], temp_df[col])
            
            # Perform Chi-Square test
            chi2, p, dof, ex = chi2_contingency(contingency_table)
            
            if chi2 > best_chi2:
                best_chi2 = chi2
                best_col = col
                
        return (best_col, best_chi2)
    
    def data_description(self, df):
        logging.info("Data Description:")
        logging.info(f"Shape of the dataset: {df.shape}")
        logging.info(f"Columns in the dataset: {df.columns.tolist()}")
        logging.info(f"Data types of columns:\n{df.dtypes}")
        logging.info(f"Number of missing values in each column (%):\n{df.isnull().sum()*100/len(df)}")
        logging.info(f"Number of duplicate rows: {df.duplicated().sum()}")
        print("Data Description:")
        print(f"Shape of the dataset: {df.shape}")
        print(f"Columns in the dataset: {df.columns.tolist()}")
        print(f"Data types of columns:\n{df.dtypes}")
        print(f"Number of missing values in each column (%):\n{df.isnull().sum()*100/len(df)}")
        print(f"Number of duplicate rows: {df.duplicated().sum()}")
    
    def handle_missing_values(self, df):
        # removing columns with more than 90% missing values
        logging.info("Handling missing values...")
        high_missing_cols = df.columns[df.isnull().mean() > 0.9]
        logging.info(f"Columns with more than 90% missing values: {high_missing_cols.tolist()}")
        df = df.drop(columns=high_missing_cols)
        
        # filling with unknown for payer_code and medical_specialty, no_test for A1Cresult  
        df['payer_code'] = df['payer_code'].fillna('UN')
        df['medical_specialty'] = df['medical_specialty'].fillna('Unknown')
        df['A1Cresult'] = df['A1Cresult'].fillna('No_test')

        logging.info("Imputed Missing values with static values for the following columns: payer_code, medical_specialty, A1Cresult")

        # mode imputation for categorical columns
        # 1. Columns with 0% missing values
        cols_0_missing = df.columns[df.isnull().mean() == 0]
        logging.info(f"Columns with 0% missing values: {cols_0_missing.tolist()}")

        # 2. Numerical columns in the dataset
        numerical_cols = ["time_in_hospital","num_lab_procedures","num_procedures","num_medications","number_outpatient","number_emergency","number_inpatient","number_diagnoses"]

        # 3. Categorical columns in the dataset
        categorical_cols = [col for col in df.columns if col not in numerical_cols]

        # 4. Impute missing values in categorical columns with chi squared imputation for finding the best partitioning of the data in cols_0_missing and categorical_cols


        missing_cols = df.columns[df.isnull().mean() > 0]

        for col in missing_cols:
            best_col, chi2_value = self.find_best_partition_col(df, col, list(set(cols_0_missing.tolist()).intersection(categorical_cols)))
            logging.info(f"For column '{col}', best partitioning column: '{best_col}' with Chi-Square value: {chi2_value}")
            df = self.impute_mode_by_group(df, col, best_col)

        logging.info("Completed handling missing values.")
        logging.info(f"Number of missing values in each column (%):\n{df.isnull().sum()*100/len(df)}")
        return df
    
    def plot_numerical_histograms(self,df, cols=3):
        """
        Generates a grid of histograms for numerical features.
        Supports 'Data Understanding' by revealing skewness and outliers.
        """
        # Select only numerical columns, excluding IDs
        numeric_cols = ["time_in_hospital","num_lab_procedures","num_procedures","num_medications","number_outpatient","number_emergency","number_inpatient","number_diagnoses"]

        
        # numeric_cols = [c for c in numeric_cols if 'id' not in c.lower() and 'nbr' not in c.lower()]
        print(f"Numerical columns for histogram plotting: {numeric_cols}")
        
        rows = math.ceil(len(numeric_cols) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            sns.histplot(df[col], kde=True, ax=axes[i], color='teal')
            axes[i].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('')
            
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 2. Ensure the directory exists
        output_dir = os.path.join("artifacts", "plots")
        os.makedirs(output_dir, exist_ok=True)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"numerical_histograms_{timestamp}.png")
        plt.savefig(plot_path, dpi=300)  # Use dpi=300 for high quality
        plt.close()  # Crucial: Close the plot to free up memory

        print(f"numerical_histograms Plot saved successfully to: {plot_path}")
        logging.info(f"numerical_histograms Plot saved successfully to: {plot_path}")

    def plot_numerical_boxplots(self, df, cols=3):
        """
        Generates a grid of box plots for numerical features.
        Used to detect outliers as part of 'Data Understanding'.
        """
        # Select numerical columns and exclude ID/administrative nbrs
        numeric_cols = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses"]
        
        rows = math.ceil(len(numeric_cols) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
            sns.boxplot(x=df[col], ax=axes[i], color='salmon', fliersize=4)
            axes[i].set_title(f'Outlier Analysis: {col}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('')
            
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 2. Ensure the directory exists
        output_dir = os.path.join("artifacts", "plots")
        os.makedirs(output_dir, exist_ok=True)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"numerical_boxplots_{timestamp}.png")
        plt.savefig(plot_path, dpi=300)  # Use dpi=300 for high quality
        plt.close()  # Crucial: Close the plot to free up memory

        print(f"numerical_boxplots Plot saved successfully to: {plot_path}")
        logging.info(f"numerical_boxplots Plot saved successfully to: {plot_path}")

    def plot_correlation_heatmap(self, df):
        """
        Generates a correlation heatmap for numerical features.
        Helps identify multicollinearity and feature relationships.
        """
        numeric_cols = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses"]

        corr_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Heatmap of Numerical Features', fontsize=14, fontweight='bold')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 2. Ensure the directory exists
        output_dir = os.path.join("artifacts", "plots")
        os.makedirs(output_dir, exist_ok=True)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"correlation_heatmap_{timestamp}.png")
        plt.savefig(plot_path, dpi=300)  # Use dpi=300 for high quality
        plt.close()  # Crucial: Close the plot to free up memory

        print(f"correlation_heatmap Plot saved successfully to: {plot_path}")
        logging.info(f"correlation_heatmap Plot saved successfully to: {plot_path}")
    
    def plot_numerical_bivariate_grid(self, df, target='readmitted', cols=3):
        """
        Generates a grid of box plots for numerical features vs. the target.
        Highlights predictive clinical signals for the Ensemble assessment.
        """
        # Select numerical columns and exclude ID/administrative nbrs
        numerical_cols = ["time_in_hospital","num_lab_procedures","num_procedures","num_medications","number_outpatient","number_emergency","number_inpatient","number_diagnoses"]
        # numeric_cols = [c for c in numerical_cols if 'id' not in c.lower() and 'nbr' not in c.lower()]
        
        rows = math.ceil(len(numerical_cols) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 5))
        axes = axes.flatten()

        for i, col in enumerate(numerical_cols):
            # Using a boxplot to compare distributions across target classes
            sns.boxplot(data=df, x=target, y=col, ax=axes[i], palette='Set2')
            
            axes[i].set_title(f'{col} vs {target}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel(target)
            axes[i].set_ylabel(col)
            
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 2. Ensure the directory exists
        output_dir = os.path.join("artifacts", "plots")
        os.makedirs(output_dir, exist_ok=True)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"numerical_bivariate_grid_{timestamp}.png")
        plt.savefig(plot_path, dpi=300)  # Use dpi=300 for high quality
        plt.close()  # Crucial: Close the plot to free up memory

        print(f"numerical_bivariate_grid Plot saved successfully to: {plot_path}")
        logging.info(f"numerical_bivariate_grid Plot saved successfully to: {plot_path}")

    def get_categorical_eda_features(self, df, threshold=900):
        """
        Selects categorical columns with unique counts below a threshold.
        Excludes unique IDs to focus on clinical and demographic signals.
        """

        candidates = list(df.columns)
        
        # Filter by unique count threshold
        eda_features = [
            col for col in candidates 
            if df[col].nunique() < threshold
        ]
        
        return eda_features

    def plot_categorical_bivariate_grid_with_pct(self,df, features, target='readmitted', cols=2):
        """
        Generates a grid of bivariate plots with explicit percentage labels on bars.
        Handles high cardinality with horizontal layouts and low cardinality with vertical.
        """
        rows = math.ceil(len(features) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 6))
        axes = axes.flatten()

        custom_palette = {"<30": "#e74c3c", ">30": "#f39c12", "NO": "#2ecc71"}
        hue_order = ["<30", ">30", "NO"]

        for i, col in enumerate(features):
            # Filter for top 15 categories to maintain professional formatting
            top_categories = df[col].value_counts().head(15).index
            df_filtered = df[df[col].isin(top_categories)]
            
            # Calculate percentage distribution
            pct_df = (df_filtered.groupby(col)[target]
                    .value_counts(normalize=True)
                    .rename('percentage')
                    .mul(100)
                    .reset_index())

            # --- LOGIC GATE FOR ORIENTATION ---
            is_high_cardinality = df[col].nunique() > 10
            
            if not is_high_cardinality:
                # Vertical Plot
                sns.barplot(data=pct_df, x=col, y='percentage', hue=target, 
                            ax=axes[i], palette=custom_palette, hue_order=hue_order)
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add labels on top of vertical bars
                for p in axes[i].patches:
                    if p.get_height() > 0:
                        axes[i].annotate(f'{p.get_height():.1f}%', 
                                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                                        ha='center', va='baseline', fontsize=9, 
                                        fontweight='bold', color='black', xytext=(0, 5),
                                        textcoords='offset points')
            else:
                # Horizontal Plot
                sns.barplot(data=pct_df, y=col, x='percentage', hue=target, 
                            ax=axes[i], palette=custom_palette, hue_order=hue_order)
                
                # Add labels to the right of horizontal bars
                for p in axes[i].patches:
                    if p.get_width() > 0:
                        axes[i].annotate(f'{p.get_width():.1f}%', 
                                        (p.get_width(), p.get_y() + p.get_height() / 2.), 
                                        ha='left', va='center', fontsize=9, 
                                        fontweight='bold', color='black', xytext=(5, 0),
                                        textcoords='offset points')

            axes[i].set_title(f'Readmission Risk by {col}', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Percentage within Category (%)')
            axes[i].set_ylabel('')
            axes[i].legend(title="Status", loc='upper right', fontsize='x-small')

        # Hide empty plots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 2. Ensure the directory exists
        output_dir = os.path.join("artifacts", "plots")
        os.makedirs(output_dir, exist_ok=True)

        # 3. Save the file instead of showing it
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"categorical_bivariate_grid_with_pct_{timestamp}.png")
        plt.savefig(plot_path, dpi=300)  # Use dpi=300 for high quality
        plt.close()  # Crucial: Close the plot to free up memory

        print(f"categorical_bivariate_grid_with_pct Plot saved successfully to: {plot_path}")
        logging.info(f"categorical_bivariate_grid_with_pct Plot saved successfully to: {plot_path}")

    def plot_categorical_grid_90pct(self,df, features, cols=3):
        """
        Generates a grid of countplots. For features with >20 unique values, 
        it only displays categories representing 90% of the data.
        """
        rows = math.ceil(len(features) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 5))
        axes = axes.flatten()

        for i, col in enumerate(features):
            # Calculate counts and percentages
            counts = df[col].value_counts(dropna=False)
            total = len(df[col])
            
            # Logic for high cardinality columns (>20 unique values)
            if df[col].nunique() > 20:
                # Calculate cumulative percentage
                pct_contributions = counts / total
                cumulative_pct = pct_contributions.cumsum()
                
                # Find categories covering 90%
                top_categories = cumulative_pct[cumulative_pct <= 0.75].index.tolist()
                
                # If 90% is just 1 item or empty, ensure at least a few items show
                if len(top_categories) < 1:
                    top_categories = counts.head(5).index.tolist()
                    
                plot_data = df[df[col].isin(top_categories)]
                title_suffix = "(Top 75% Signal)"
            else:
                plot_data = df
                title_suffix = ""

            # Create the plot
            sns.countplot(
                data=plot_data, 
                x=col, 
                ax=axes[i], 
                palette='viridis',
                order=counts.index[:len(plot_data[col].unique())] if df[col].nunique() > 20 else counts.index
            )
            
            # Add Percentage Labels
            current_plot_total = len(df) # Always calculate % against the WHOLE dataset
            for p in axes[i].patches:
                label = f'{100 * p.get_height() / current_plot_total:.1f}%'
                axes[i].annotate(label, (p.get_x() + p.get_width() / 2, p.get_height()),
                                ha='center', va='bottom', fontsize=9, fontweight='bold')

            axes[i].set_title(f'{col} {title_suffix}', fontsize=12, fontweight='bold')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('Count')

        # Clean up empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 2. Ensure the directory exists
        output_dir = os.path.join("artifacts", "plots")
        os.makedirs(output_dir, exist_ok=True)

        # 3. Save the file instead of showing it
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"categorical_grid_90pct_{timestamp}.png")
        plt.savefig(plot_path, dpi=300)  # Use dpi=300 for high quality
        plt.close()  # Crucial: Close the plot to free up memory

        print(f"categorical_grid_90pct Plot saved successfully to: {plot_path}")

        logging.info(f"categorical_grid_90pct Plot saved successfully to: {plot_path}")

        # plt.tight_layout()
        # plt.show()

    

    def initiate_data_ingestion(self, data_file = 'diabetic_data.csv'):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(os.path.join('data', data_file))
            logging.info("Read the dataset as dataframe")
            self.data_description(df)


            cols = df.columns 

            for col in cols:
                df[col] = df[col].apply(self.convert_to_none)

            logging.info("Converted '?' to NaN in the dataset")

            # drop id columns
            df = df.drop(columns=['encounter_id', 'patient_nbr'])
            logging.info("Dropped id columns: encounter_id and patient_nbr")

            df = self.handle_missing_values(df)

            logging.info("Handled missing values in the dataset")


            # handling duplicates
            num_duplicates = df.duplicated().sum()
            logging.info(f"Number of duplicate rows in the dataset: {num_duplicates}")
            df = df.drop_duplicates()
            logging.info(f"Number of rows after removing duplicates: {len(df)}")

            # Administrative IDs should be cast to string so they aren't treated as continuous numbers
            admin_ids = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
            for col in admin_ids:
                if col in df.columns:
                    df[col] = df[col].astype(str)

            # handling outliers in numerical columns by capping at 99th percentile using winsorization to preserve data distribution while mitigating extreme values that could skew the model training.
            
            df['num_lab_procedures'] = winsorize(df['num_lab_procedures'], limits=[0.01, 0.01])
            df['num_medications'] = winsorize(df['num_medications'], limits=[0.01, 0.01])
            df['time_in_hospital'] = winsorize(df['time_in_hospital'], limits=[0.01, 0.01])

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data is saved")

            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42, stratify=df['readmitted'])
            val_set, test_set = train_test_split(test_set, test_size=0.5, random_state=42, stratify=test_set['readmitted'])

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            val_set.to_csv(self.ingestion_config.val_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            numerical_cols = ["time_in_hospital","num_lab_procedures","num_procedures","num_medications","number_outpatient","number_emergency","number_inpatient","number_diagnoses"]
            self.plot_categorical_grid_90pct(df, features=list(set(self.get_categorical_eda_features(df))-set(numerical_cols)), cols=3)
            self.plot_numerical_histograms(df)
            self.plot_numerical_boxplots(df)
            self.plot_correlation_heatmap(df)
            self.plot_numerical_bivariate_grid(df, target='readmitted')
            self.plot_categorical_bivariate_grid_with_pct(df, features=list(set(self.get_categorical_eda_features(df))-set(numerical_cols)), target='readmitted')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.val_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data, val_data, test_data = data_ingestion.initiate_data_ingestion(data_file='diabetic_data.csv')
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, val_data, test_data)