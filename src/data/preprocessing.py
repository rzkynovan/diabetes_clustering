"""
Preprocessing Module for Diabetes Clustering Project
Author: Novan

This module contains functions for data cleaning, missing value handling,
and data transformation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiabetesPreprocessor:
    """
    Comprehensive preprocessor for diabetes dataset
    """
    
    def __init__(self, config: Dict):
        """
        Initialize preprocessor with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.feature_names = {}
        self.statistics = {}
        
    def identify_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Identify and categorize columns
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary of categorized columns
        """
        logger.info("Identifying column types...")
        
        columns = {
            'id_columns': [],
            'target': self.config['data']['target_column'],
            'numerical': [],
            'categorical': [],
            'binary': [],
            'high_missing': [],
            'medication': []
        }
        
        # Identify ID columns
        id_keywords = ['id', 'nbr', 'encounter']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in id_keywords):
                columns['id_columns'].append(col)
        
        # Identify high missing columns (>80%)
        missing_pct = df.isnull().sum() / len(df) * 100
        columns['high_missing'] = missing_pct[missing_pct > 80].index.tolist()
        
        # Identify medication columns
        med_keywords = ['metformin', 'insulin', 'glyburide', 'glipizide', 'glimepiride',
                       'pioglitazone', 'rosiglitazone', 'repaglinide', 'nateglinide',
                       'chlorpropamide', 'tolbutamide', 'acarbose', 'miglitol',
                       'troglitazone', 'tolazamide', 'examide', 'citoglipton',
                       'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
                       'metformin-rosiglitazone', 'metformin-pioglitazone']
        
        for col in df.columns:
            if any(med in col.lower() for med in med_keywords):
                columns['medication'].append(col)
        
        # Identify numerical, categorical, and binary columns
        for col in df.columns:
            if col in columns['id_columns'] or col in columns['high_missing']:
                continue
            if col == columns['target']:
                continue
            if col in columns['medication']:
                continue
                
            if df[col].dtype in ['int64', 'float64']:
                # Check if binary
                unique_vals = df[col].nunique()
                if unique_vals == 2:
                    columns['binary'].append(col)
                else:
                    columns['numerical'].append(col)
            else:
                columns['categorical'].append(col)
        
        # Log summary
        logger.info(f"Column categorization:")
        logger.info(f"  - ID columns: {len(columns['id_columns'])}")
        logger.info(f"  - High missing (>80%): {len(columns['high_missing'])}")
        logger.info(f"  - Numerical: {len(columns['numerical'])}")
        logger.info(f"  - Categorical: {len(columns['categorical'])}")
        logger.info(f"  - Binary: {len(columns['binary'])}")
        logger.info(f"  - Medication: {len(columns['medication'])}")
        
        return columns
    
    def remove_high_missing_columns(self, df: pd.DataFrame, 
                                   threshold: float = 0.8) -> pd.DataFrame:
        """
        Remove columns with high percentage of missing values
        
        Args:
            df: Input dataframe
            threshold: Missing value threshold (default 0.8 = 80%)
            
        Returns:
            Dataframe with high-missing columns removed
        """
        missing_pct = df.isnull().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        
        if cols_to_drop:
            logger.info(f"Removing {len(cols_to_drop)} columns with >{threshold*100}% missing:")
            for col in cols_to_drop:
                logger.info(f"  - {col}: {missing_pct[col]*100:.2f}% missing")
            
            df = df.drop(columns=cols_to_drop)
        else:
            logger.info(f"No columns with >{threshold*100}% missing values")
        
        self.statistics['removed_high_missing'] = cols_to_drop
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, 
                            columns: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Handle missing values with appropriate strategies
        
        Args:
            df: Input dataframe
            columns: Dictionary of categorized columns
            
        Returns:
            Dataframe with missing values handled
        """
        logger.info("Handling missing values...")
        df = df.copy()
        
        missing_summary = []
        
        # Handle weight (if present)
        if 'weight' in df.columns and df['weight'].isnull().any():
            logger.info("Handling 'weight' with age-stratified median imputation")
            
            # Create missing indicator
            df['weight_missing'] = df['weight'].isnull().astype(int)
            
            # Age-stratified imputation
            if 'age' in df.columns:
                for age_group in df['age'].unique():
                    mask = (df['age'] == age_group) & df['weight'].isnull()
                    if mask.any():
                        median_val = df[df['age'] == age_group]['weight'].median()
                        if pd.notna(median_val):
                            df.loc[mask, 'weight'] = median_val
                        else:
                            # If no data for this age group, use overall median
                            df.loc[mask, 'weight'] = df['weight'].median()
            else:
                # No age column, use overall median
                df['weight'].fillna(df['weight'].median(), inplace=True)
            
            missing_summary.append({
                'column': 'weight',
                'strategy': 'age-stratified median + missing indicator'
            })
        
        # Handle payer_code
        if 'payer_code' in df.columns and df['payer_code'].isnull().any():
            logger.info("Handling 'payer_code' with 'Unknown' category")
            df['payer_code'].fillna('Unknown', inplace=True)
            missing_summary.append({
                'column': 'payer_code',
                'strategy': 'Unknown category'
            })
        
        # Handle medical_specialty
        if 'medical_specialty' in df.columns and df['medical_specialty'].isnull().any():
            logger.info("Handling 'medical_specialty' with 'Unknown' category")
            df['medical_specialty_missing'] = df['medical_specialty'].isnull().astype(int)
            df['medical_specialty'].fillna('Unknown', inplace=True)
            missing_summary.append({
                'column': 'medical_specialty',
                'strategy': 'Unknown category + missing indicator'
            })
        
        # Handle race
        if 'race' in df.columns and df['race'].isnull().any():
            logger.info("Handling 'race' with mode imputation")
            mode_val = df['race'].mode()[0] if not df['race'].mode().empty else 'Unknown'
            df['race'].fillna(mode_val, inplace=True)
            missing_summary.append({
                'column': 'race',
                'strategy': f'mode imputation ({mode_val})'
            })
        
        # Handle other categorical columns
        for col in columns['categorical']:
            if col in df.columns and df[col].isnull().any():
                missing_count = df[col].isnull().sum()
                missing_pct = missing_count / len(df) * 100
                
                if missing_pct < 5:
                    # Low missing: use mode
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col].fillna(mode_val, inplace=True)
                    logger.info(f"  - {col}: mode imputation ({missing_pct:.2f}% missing)")
                    missing_summary.append({
                        'column': col,
                        'strategy': f'mode ({mode_val})'
                    })
                else:
                    # Higher missing: use Unknown + indicator
                    df[f'{col}_missing'] = df[col].isnull().astype(int)
                    df[col].fillna('Unknown', inplace=True)
                    logger.info(f"  - {col}: Unknown + indicator ({missing_pct:.2f}% missing)")
                    missing_summary.append({
                        'column': col,
                        'strategy': 'Unknown + missing indicator'
                    })
        
        # Handle numerical columns
        for col in columns['numerical']:
            if col in df.columns and df[col].isnull().any():
                missing_count = df[col].isnull().sum()
                missing_pct = missing_count / len(df) * 100
                
                # Use median for numerical
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"  - {col}: median imputation ({missing_pct:.2f}% missing)")
                missing_summary.append({
                    'column': col,
                    'strategy': f'median ({median_val:.2f})'
                })
        
        # Handle medication columns (categorical: No/Steady/Up/Down)
        for col in columns['medication']:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna('No', inplace=True)
                logger.info(f"  - {col}: filled with 'No'")
                missing_summary.append({
                    'column': col,
                    'strategy': 'No (no medication)'
                })
        
        self.statistics['missing_value_handling'] = missing_summary
        
        # Final check
        remaining_missing = df.isnull().sum().sum()
        logger.info(f"Remaining missing values: {remaining_missing}")
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame,
                                   columns: Dict[str, List[str]],
                                   method: str = 'auto') -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df: Input dataframe
            columns: Dictionary of categorized columns
            method: Encoding method ('onehot', 'label', 'target', 'auto')
            
        Returns:
            Dataframe with encoded features
        """
        logger.info("Encoding categorical features...")
        df = df.copy()
        
        encoding_summary = []
        
        for col in columns['categorical']:
            if col not in df.columns:
                continue
            
            n_unique = df[col].nunique()
            
            # Determine encoding strategy
            if method == 'auto':
                if n_unique <= 10:
                    encoding = 'onehot'
                elif n_unique <= 50:
                    encoding = 'label'  # Will use target encoding later if needed
                else:
                    encoding = 'label'
            else:
                encoding = method
            
            # Apply encoding
            if encoding == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
                
                logger.info(f"  - {col}: one-hot encoded ({n_unique} categories → {len(dummies.columns)} features)")
                encoding_summary.append({
                    'column': col,
                    'method': 'onehot',
                    'n_categories': n_unique,
                    'n_features': len(dummies.columns)
                })
                
            elif encoding == 'label':
                # Label encoding (ordinal)
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
                
                logger.info(f"  - {col}: label encoded ({n_unique} categories)")
                encoding_summary.append({
                    'column': col,
                    'method': 'label',
                    'n_categories': n_unique
                })
        
        # Handle medication columns (convert to numeric)
        medication_encoding = {'No': 0, 'Steady': 1, 'Down': 2, 'Up': 3}
        
        for col in columns['medication']:
            if col in df.columns:
                df[col] = df[col].map(medication_encoding)
                logger.info(f"  - {col}: medication encoding (No=0, Steady=1, Down=2, Up=3)")
                encoding_summary.append({
                    'column': col,
                    'method': 'medication_specific',
                    'mapping': medication_encoding
                })
        
        self.statistics['encoding'] = encoding_summary
        return df
    
    def scale_features(self, df: pd.DataFrame,
                      columns: Dict[str, List[str]],
                      feature_groups: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
        """
        Scale numerical features with appropriate scalers
        
        Args:
            df: Input dataframe
            columns: Dictionary of categorized columns
            feature_groups: Optional dict mapping feature groups to column names
            
        Returns:
            Dataframe with scaled features
        """
        logger.info("Scaling numerical features...")
        df = df.copy()
        
        if feature_groups is None:
            # Default scaling: RobustScaler for all numerical
            numerical_cols = [col for col in columns['numerical'] if col in df.columns]
            
            if numerical_cols:
                scaler = RobustScaler()
                df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
                self.scalers['numerical'] = scaler
                logger.info(f"  - Applied RobustScaler to {len(numerical_cols)} numerical features")
        else:
            # Custom scaling per feature group
            for group_name, cols in feature_groups.items():
                cols = [col for col in cols if col in df.columns]
                if not cols:
                    continue
                
                # Determine scaler type
                if 'clinical' in group_name.lower():
                    scaler = StandardScaler()
                    scaler_name = 'StandardScaler'
                elif 'count' in group_name.lower() or 'num_' in group_name.lower():
                    scaler = MinMaxScaler()
                    scaler_name = 'MinMaxScaler'
                else:
                    scaler = RobustScaler()
                    scaler_name = 'RobustScaler'
                
                df[cols] = scaler.fit_transform(df[cols])
                self.scalers[group_name] = scaler
                logger.info(f"  - Applied {scaler_name} to {len(cols)} features in '{group_name}'")
        
        return df
    
    def split_data(self, df: pd.DataFrame, 
                  stratify_col: Optional[str] = None,
                  train_size: float = 0.7,
                  val_size: float = 0.15,
                  test_size: float = 0.15,
                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets
        
        Args:
            df: Input dataframe
            stratify_col: Column to stratify on
            train_size: Training set proportion
            val_size: Validation set proportion
            test_size: Test set proportion
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Splitting data...")
        
        # Verify proportions sum to 1
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
            "train_size + val_size + test_size must equal 1.0"
        
        # Prepare stratification
        stratify_array = df[stratify_col] if stratify_col else None
        
        # First split: train + (val+test)
        train_df, temp_df = train_test_split(
            df,
            train_size=train_size,
            random_state=random_state,
            stratify=stratify_array
        )
        
        # Second split: val + test
        val_proportion = val_size / (val_size + test_size)
        stratify_temp = temp_df[stratify_col] if stratify_col else None
        
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_proportion,
            random_state=random_state,
            stratify=stratify_temp
        )
        
        logger.info(f"Data split:")
        logger.info(f"  - Train: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
        logger.info(f"  - Val:   {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
        logger.info(f"  - Test:  {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
        
        if stratify_col:
            logger.info(f"\nStratification check ({stratify_col}):")
            for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
                dist = split_df[stratify_col].value_counts(normalize=True) * 100
                logger.info(f"  {split_name}:")
                for val, pct in dist.items():
                    logger.info(f"    - {val}: {pct:.2f}%")
        
        self.statistics['data_split'] = {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'stratify_column': stratify_col
        }
        
        return train_df, val_df, test_df
    
    def get_preprocessing_summary(self) -> Dict:
        """
        Get summary of all preprocessing steps
        
        Returns:
            Dictionary containing preprocessing statistics
        """
        return self.statistics


def create_feature_groups(df: pd.DataFrame, config: Dict) -> Dict[str, List[str]]:
    """
    Create feature groups for multi-perspective clustering
    
    Args:
        df: Preprocessed dataframe
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping perspective names to feature lists
    """
    logger.info("Creating feature groups for multi-perspective clustering...")
    
    # Get configured feature lists
    clinical_keywords = config.get('features', {}).get('clinical_features', [])
    treatment_keywords = config.get('features', {}).get('treatment_features', [])
    admin_keywords = config.get('features', {}).get('administrative_features', [])
    
    feature_groups = {
        'clinical': [],
        'treatment': [],
        'administrative': [],
        'demographic': []
    }
    
    # Categorize each column
    for col in df.columns:
        col_lower = col.lower()
        
        # Skip target and ID columns
        if col == config['data']['target_column']:
            continue
        if any(keyword in col_lower for keyword in ['id', 'nbr', 'encounter']):
            continue
        
        # Categorize
        if any(keyword in col_lower for keyword in ['age', 'gender', 'race']):
            feature_groups['demographic'].append(col)
        elif any(keyword in col_lower for keyword in clinical_keywords):
            feature_groups['clinical'].append(col)
        elif any(med in col_lower for med in treatment_keywords):
            feature_groups['treatment'].append(col)
        elif any(keyword in col_lower for keyword in admin_keywords):
            feature_groups['administrative'].append(col)
        else:
            # Default to clinical if unclear
            feature_groups['clinical'].append(col)
    
    # Log summary
    for group, features in feature_groups.items():
        logger.info(f"  - {group.capitalize()}: {len(features)} features")
    
    return feature_groups
