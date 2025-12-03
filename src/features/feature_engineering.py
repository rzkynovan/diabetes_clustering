"""
Feature Engineering Module for Diabetes Clustering Project
Author: Novan

This module contains functions for creating domain-specific features
for diabetes patient clustering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiabetesFeatureEngineer:
    """
    Feature engineering for diabetes dataset
    """
    
    def __init__(self, config: Dict):
        """
        Initialize feature engineer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.created_features = []
        
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with engineered features
        """
        logger.info("="*70)
        logger.info("FEATURE ENGINEERING STARTED")
        logger.info("="*70)
        
        df = df.copy()
        initial_features = df.shape[1]
        
        # Clinical features
        df = self.create_clinical_features(df)
        
        # Treatment features
        df = self.create_treatment_features(df)
        
        # Administrative features
        df = self.create_administrative_features(df)
        
        # Demographic features
        df = self.create_demographic_features(df)
        
        # Interaction features
        df = self.create_interaction_features(df)
        
        final_features = df.shape[1]
        new_features = final_features - initial_features
        
        logger.info("="*70)
        logger.info(f"FEATURE ENGINEERING COMPLETED")
        logger.info(f"  - Initial features: {initial_features}")
        logger.info(f"  - Final features: {final_features}")
        logger.info(f"  - New features created: {new_features}")
        logger.info(f"  - Total created: {len(self.created_features)}")
        logger.info("="*70)
        
        return df
    
    def create_clinical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create clinical domain features
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with clinical features
        """
        logger.info("\n1. Creating Clinical Features...")
        df = df.copy()
        features_created = []
        
        # Comorbidity features
        if 'number_diagnoses' in df.columns:
            # Comorbidity score (simple count)
            df['comorbidity_score'] = df['number_diagnoses']
            features_created.append('comorbidity_score')
            
            # Comorbidity complexity categories
            df['comorbidity_low'] = (df['number_diagnoses'] <= 3).astype(int)
            df['comorbidity_medium'] = ((df['number_diagnoses'] > 3) & 
                                        (df['number_diagnoses'] <= 6)).astype(int)
            df['comorbidity_high'] = (df['number_diagnoses'] > 6).astype(int)
            features_created.extend(['comorbidity_low', 'comorbidity_medium', 'comorbidity_high'])
        
        # Diagnosis diversity
        diagnosis_cols = [col for col in df.columns if 'diag_' in col]
        if len(diagnosis_cols) >= 2:
            # Has secondary diagnosis
            if 'diag_2' in df.columns:
                df['has_secondary_diagnosis'] = df['diag_2'].notna().astype(int)
                features_created.append('has_secondary_diagnosis')
            
            # Has tertiary diagnosis
            if 'diag_3' in df.columns:
                df['has_tertiary_diagnosis'] = df['diag_3'].notna().astype(int)
                features_created.append('has_tertiary_diagnosis')
            
            # Multiple diagnoses
            df['multiple_diagnoses'] = (df['has_secondary_diagnosis'] + 
                                       df.get('has_tertiary_diagnosis', 0))
            features_created.append('multiple_diagnoses')
        
        # Lab procedure intensity
        if 'num_lab_procedures' in df.columns:
            df['lab_intensity_low'] = (df['num_lab_procedures'] <= 40).astype(int)
            df['lab_intensity_medium'] = ((df['num_lab_procedures'] > 40) & 
                                          (df['num_lab_procedures'] <= 70)).astype(int)
            df['lab_intensity_high'] = (df['num_lab_procedures'] > 70).astype(int)
            features_created.extend(['lab_intensity_low', 'lab_intensity_medium', 'lab_intensity_high'])
        
        # Clinical complexity score (composite)
        clinical_components = []
        if 'num_procedures' in df.columns:
            clinical_components.append(df['num_procedures'])
        if 'num_lab_procedures' in df.columns:
            clinical_components.append(df['num_lab_procedures'] / 10)  # Normalize scale
        if 'number_diagnoses' in df.columns:
            clinical_components.append(df['number_diagnoses'])
        
        if clinical_components:
            df['clinical_complexity_score'] = sum(clinical_components)
            features_created.append('clinical_complexity_score')
        
        # Procedure to lab ratio
        if 'num_procedures' in df.columns and 'num_lab_procedures' in df.columns:
            df['procedure_lab_ratio'] = (df['num_procedures'] / 
                                         (df['num_lab_procedures'] + 1))  # Avoid division by zero
            features_created.append('procedure_lab_ratio')
        
        logger.info(f"   Created {len(features_created)} clinical features:")
        for feat in features_created:
            logger.info(f"     - {feat}")
        
        self.created_features.extend(features_created)
        return df
    
    def create_treatment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create treatment pattern features
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with treatment features
        """
        logger.info("\n2. Creating Treatment Features...")
        df = df.copy()
        features_created = []
        
        # Identify medication columns
        med_keywords = ['metformin', 'insulin', 'glyburide', 'glipizide', 'glimepiride',
                       'pioglitazone', 'rosiglitazone', 'repaglinide', 'nateglinide',
                       'chlorpropamide', 'tolbutamide', 'acarbose', 'miglitol',
                       'troglitazone', 'tolazamide', 'examide', 'citoglipton',
                       'glyburide-metformin', 'glipizide-metformin', 
                       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
                       'metformin-pioglitazone']
        
        medication_cols = [col for col in df.columns 
                          if any(med in col.lower() for med in med_keywords)]
        
        if medication_cols:
            # Total medications count (assuming already encoded: 0=No, 1=Steady, 2=Down, 3=Up)
            # Count non-zero values
            df['total_medications'] = (df[medication_cols] > 0).sum(axis=1)
            features_created.append('total_medications')
            
            # Medication changes count
            # Up (3) or Down (2) indicate changes
            df['medication_changes_count'] = ((df[medication_cols] == 2) | 
                                             (df[medication_cols] == 3)).sum(axis=1)
            features_created.append('medication_changes_count')
            
            # Medication up count (intensification)
            df['medication_up_count'] = (df[medication_cols] == 3).sum(axis=1)
            features_created.append('medication_up_count')
            
            # Medication down count (reduction)
            df['medication_down_count'] = (df[medication_cols] == 2).sum(axis=1)
            features_created.append('medication_down_count')
            
            # Poly-pharmacy flag (>5 medications)
            df['poly_pharmacy'] = (df['total_medications'] > 5).astype(int)
            features_created.append('poly_pharmacy')
            
            # Medication intensity categories
            df['medication_intensity_low'] = (df['total_medications'] <= 2).astype(int)
            df['medication_intensity_medium'] = ((df['total_medications'] > 2) & 
                                                 (df['total_medications'] <= 5)).astype(int)
            df['medication_intensity_high'] = (df['total_medications'] > 5).astype(int)
            features_created.extend(['medication_intensity_low', 
                                    'medication_intensity_medium',
                                    'medication_intensity_high'])
        
        # Insulin-specific features
        if 'insulin' in df.columns:
            df['insulin_dependent'] = (df['insulin'] > 0).astype(int)
            df['insulin_increased'] = (df['insulin'] == 3).astype(int)
            features_created.extend(['insulin_dependent', 'insulin_increased'])
        
        # Metformin-specific features
        if 'metformin' in df.columns:
            df['on_metformin'] = (df['metformin'] > 0).astype(int)
            features_created.append('on_metformin')
        
        # Oral agents only (no insulin)
        if 'insulin_dependent' in df.columns and 'total_medications' in df.columns:
            df['oral_agents_only'] = ((df['total_medications'] > 0) & 
                                     (df['insulin_dependent'] == 0)).astype(int)
            features_created.append('oral_agents_only')
        
        # Medication diversity (entropy-like measure)
        if medication_cols:
            # Count unique medication patterns
            med_variety = df[medication_cols].nunique(axis=1)
            df['medication_diversity'] = med_variety
            features_created.append('medication_diversity')
        
        # Diabetes medication prescribed
        if 'diabetesMed' in df.columns:
            # Assuming it's Yes/No, encode if needed
            if df['diabetesMed'].dtype == 'object':
                df['diabetesMed_binary'] = (df['diabetesMed'] == 'Yes').astype(int)
                features_created.append('diabetesMed_binary')
        
        # Medication change flag
        if 'change' in df.columns:
            if df['change'].dtype == 'object':
                df['medication_changed'] = (df['change'] == 'Ch').astype(int)
                features_created.append('medication_changed')
        
        # Treatment complexity score
        treatment_components = []
        if 'total_medications' in df.columns:
            treatment_components.append(df['total_medications'])
        if 'medication_changes_count' in df.columns:
            treatment_components.append(df['medication_changes_count'] * 2)  # Weight changes more
        if 'insulin_dependent' in df.columns:
            treatment_components.append(df['insulin_dependent'] * 3)  # Insulin is significant
        
        if treatment_components:
            df['treatment_complexity_score'] = sum(treatment_components)
            features_created.append('treatment_complexity_score')
        
        logger.info(f"   Created {len(features_created)} treatment features:")
        for feat in features_created:
            logger.info(f"     - {feat}")
        
        self.created_features.extend(features_created)
        return df
    
    def create_administrative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create administrative and healthcare utilization features
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with administrative features
        """
        logger.info("\n3. Creating Administrative Features...")
        df = df.copy()
        features_created = []
        
        # Length of stay categories
        if 'time_in_hospital' in df.columns:
            df['los_short'] = (df['time_in_hospital'] <= 3).astype(int)
            df['los_medium'] = ((df['time_in_hospital'] > 3) & 
                               (df['time_in_hospital'] <= 7)).astype(int)
            df['los_long'] = (df['time_in_hospital'] > 7).astype(int)
            df['los_very_long'] = (df['time_in_hospital'] > 14).astype(int)
            features_created.extend(['los_short', 'los_medium', 'los_long', 'los_very_long'])
        
        # Total healthcare visits
        visit_cols = ['number_outpatient', 'number_emergency', 'number_inpatient']
        existing_visit_cols = [col for col in visit_cols if col in df.columns]
        
        if existing_visit_cols:
            df['total_visits'] = df[existing_visit_cols].sum(axis=1)
            features_created.append('total_visits')
            
            # Frequent visitor flag
            df['frequent_visitor'] = (df['total_visits'] > 3).astype(int)
            features_created.append('frequent_visitor')
            
            # Visit diversity (visited multiple types)
            df['visit_diversity'] = (df[existing_visit_cols] > 0).sum(axis=1)
            features_created.append('visit_diversity')
        
        # Emergency visit flag
        if 'number_emergency' in df.columns:
            df['has_emergency_visit'] = (df['number_emergency'] > 0).astype(int)
            df['multiple_emergency'] = (df['number_emergency'] > 1).astype(int)
            features_created.extend(['has_emergency_visit', 'multiple_emergency'])
        
        # Outpatient visit flag
        if 'number_outpatient' in df.columns:
            df['has_outpatient_visit'] = (df['number_outpatient'] > 0).astype(int)
            features_created.append('has_outpatient_visit')
        
        # Inpatient history
        if 'number_inpatient' in df.columns:
            df['has_inpatient_history'] = (df['number_inpatient'] > 0).astype(int)
            df['frequent_inpatient'] = (df['number_inpatient'] > 2).astype(int)
            features_created.extend(['has_inpatient_history', 'frequent_inpatient'])
        
        # Admission type analysis
        if 'admission_type_id' in df.columns:
            # Common patterns: 1=Emergency, 2=Urgent, 3=Elective
            df['emergency_admission'] = (df['admission_type_id'] == 1).astype(int)
            df['urgent_admission'] = (df['admission_type_id'] == 2).astype(int)
            df['elective_admission'] = (df['admission_type_id'] == 3).astype(int)
            features_created.extend(['emergency_admission', 'urgent_admission', 'elective_admission'])
        
        # Discharge disposition analysis
        if 'discharge_disposition_id' in df.columns:
            # Common: 1=Home, 2=Transfer to short-term hospital, 3=Transfer to SNF
            df['discharged_home'] = (df['discharge_disposition_id'] == 1).astype(int)
            df['transferred'] = (df['discharge_disposition_id'].isin([2, 3, 4, 5])).astype(int)
            features_created.extend(['discharged_home', 'transferred'])
        
        # Admission source
        if 'admission_source_id' in df.columns:
            # Common: 7=Emergency room, 1=Physician referral
            df['from_emergency'] = (df['admission_source_id'] == 7).astype(int)
            df['from_referral'] = (df['admission_source_id'] == 1).astype(int)
            features_created.extend(['from_emergency', 'from_referral'])
        
        # Resource utilization score (composite)
        resource_components = []
        if 'time_in_hospital' in df.columns:
            resource_components.append(df['time_in_hospital'])
        if 'num_procedures' in df.columns:
            resource_components.append(df['num_procedures'] * 2)
        if 'num_lab_procedures' in df.columns:
            resource_components.append(df['num_lab_procedures'] / 10)
        if 'num_medications' in df.columns:
            resource_components.append(df['num_medications'])
        
        if resource_components:
            df['resource_utilization_score'] = sum(resource_components)
            features_created.append('resource_utilization_score')
        
        # Utilization intensity categories
        if 'resource_utilization_score' in df.columns:
            df['utilization_low'] = (df['resource_utilization_score'] <= 
                                    df['resource_utilization_score'].quantile(0.33)).astype(int)
            df['utilization_high'] = (df['resource_utilization_score'] >= 
                                     df['resource_utilization_score'].quantile(0.67)).astype(int)
            features_created.extend(['utilization_low', 'utilization_high'])
        
        logger.info(f"   Created {len(features_created)} administrative features:")
        for feat in features_created:
            logger.info(f"     - {feat}")
        
        self.created_features.extend(features_created)
        return df
    
    def create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create demographic features
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with demographic features
        """
        logger.info("\n4. Creating Demographic Features...")
        df = df.copy()
        features_created = []
        
        # Age features
        if 'age' in df.columns:
            # Convert age ranges to numeric midpoint if still categorical
            if df['age'].dtype == 'object':
                age_mapping = {
                    '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
                    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
                    '[80-90)': 85, '[90-100)': 95
                }
                df['age_numeric'] = df['age'].map(age_mapping)
                features_created.append('age_numeric')
            else:
                df['age_numeric'] = df['age']
            
            # Age categories
            df['age_young'] = (df['age_numeric'] < 40).astype(int)
            df['age_middle'] = ((df['age_numeric'] >= 40) & 
                               (df['age_numeric'] < 60)).astype(int)
            df['age_senior'] = ((df['age_numeric'] >= 60) & 
                               (df['age_numeric'] < 75)).astype(int)
            df['age_elderly'] = (df['age_numeric'] >= 75).astype(int)
            features_created.extend(['age_young', 'age_middle', 'age_senior', 'age_elderly'])
        
        # Gender binary (if not already)
        if 'gender' in df.columns:
            if df['gender'].dtype == 'object':
                df['gender_male'] = (df['gender'] == 'Male').astype(int)
                df['gender_female'] = (df['gender'] == 'Female').astype(int)
                features_created.extend(['gender_male', 'gender_female'])
        
        # Race/ethnicity encoding (if categorical)
        if 'race' in df.columns:
            if df['race'].dtype == 'object':
                # Create binary flags for major categories
                df['race_caucasian'] = (df['race'] == 'Caucasian').astype(int)
                df['race_african_american'] = (df['race'] == 'AfricanAmerican').astype(int)
                df['race_hispanic'] = (df['race'] == 'Hispanic').astype(int)
                df['race_asian'] = (df['race'] == 'Asian').astype(int)
                df['race_other'] = (~df['race'].isin(['Caucasian', 'AfricanAmerican', 
                                                      'Hispanic', 'Asian'])).astype(int)
                features_created.extend(['race_caucasian', 'race_african_american',
                                        'race_hispanic', 'race_asian', 'race_other'])
        
        logger.info(f"   Created {len(features_created)} demographic features:")
        for feat in features_created:
            logger.info(f"     - {feat}")
        
        self.created_features.extend(features_created)
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with interaction features
        """
        logger.info("\n5. Creating Interaction Features...")
        df = df.copy()
        features_created = []
        
        # Age × Comorbidity interaction
        if 'age_numeric' in df.columns and 'comorbidity_score' in df.columns:
            df['age_comorbidity_interaction'] = (df['age_numeric'] / 100) * df['comorbidity_score']
            features_created.append('age_comorbidity_interaction')
        
        # Age × Medication interaction
        if 'age_numeric' in df.columns and 'total_medications' in df.columns:
            df['age_medication_interaction'] = (df['age_numeric'] / 100) * df['total_medications']
            features_created.append('age_medication_interaction')
        
        # Insulin × Age interaction
        if 'insulin_dependent' in df.columns and 'age_numeric' in df.columns:
            df['insulin_age_interaction'] = df['insulin_dependent'] * (df['age_numeric'] / 100)
            features_created.append('insulin_age_interaction')
        
        # LOS × Procedures interaction
        if 'time_in_hospital' in df.columns and 'num_procedures' in df.columns:
            df['los_procedures_interaction'] = df['time_in_hospital'] * df['num_procedures']
            features_created.append('los_procedures_interaction')
        
        # Medications × Comorbidity
        if 'total_medications' in df.columns and 'comorbidity_score' in df.columns:
            df['medication_comorbidity_interaction'] = (df['total_medications'] * 
                                                        df['comorbidity_score'])
            features_created.append('medication_comorbidity_interaction')
        
        # Emergency × Age
        if 'has_emergency_visit' in df.columns and 'age_numeric' in df.columns:
            df['emergency_age_interaction'] = df['has_emergency_visit'] * (df['age_numeric'] / 100)
            features_created.append('emergency_age_interaction')
        
        # Complex patient flag (multiple risk factors)
        risk_factors = []
        if 'age_elderly' in df.columns:
            risk_factors.append(df['age_elderly'])
        if 'poly_pharmacy' in df.columns:
            risk_factors.append(df['poly_pharmacy'])
        if 'comorbidity_high' in df.columns:
            risk_factors.append(df['comorbidity_high'])
        if 'frequent_visitor' in df.columns:
            risk_factors.append(df['frequent_visitor'])
        
        if risk_factors:
            df['complex_patient_score'] = sum(risk_factors)
            df['complex_patient'] = (df['complex_patient_score'] >= 2).astype(int)
            features_created.extend(['complex_patient_score', 'complex_patient'])
        
        logger.info(f"   Created {len(features_created)} interaction features:")
        for feat in features_created:
            logger.info(f"     - {feat}")
        
        self.created_features.extend(features_created)
        return df
    
    def get_created_features(self) -> List[str]:
        """
        Get list of all created features
        
        Returns:
            List of feature names
        """
        return self.created_features
    
    def get_feature_summary(self) -> Dict:
        """
        Get summary of feature engineering
        
        Returns:
            Dictionary with feature engineering summary
        """
        summary = {
            'total_features_created': len(self.created_features),
            'feature_list': self.created_features,
            'feature_groups': {
                'clinical': [f for f in self.created_features 
                           if any(kw in f for kw in ['comorbidity', 'diagnosis', 'lab', 'clinical', 'procedure'])],
                'treatment': [f for f in self.created_features 
                            if any(kw in f for kw in ['medication', 'insulin', 'metformin', 'treatment', 'diabetes'])],
                'administrative': [f for f in self.created_features 
                                 if any(kw in f for kw in ['los', 'visit', 'admission', 'discharge', 'utilization', 'resource'])],
                'demographic': [f for f in self.created_features 
                              if any(kw in f for kw in ['age', 'gender', 'race'])],
                'interaction': [f for f in self.created_features 
                              if 'interaction' in f or 'complex' in f]
            }
        }
        
        return summary


def engineer_features(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to engineer all features
    
    Args:
        df: Input dataframe
        config: Configuration dictionary
        
    Returns:
        Tuple of (engineered dataframe, summary dictionary)
    """
    engineer = DiabetesFeatureEngineer(config)
    df_engineered = engineer.create_all_features(df)
    summary = engineer.get_feature_summary()
    
    return df_engineered, summary
