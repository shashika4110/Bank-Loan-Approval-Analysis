import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def clean_data(df, fill_values=None):
    """
    Cleans the raw loan dataset.
    - Handles missing values.
    - Handles data types.
    """
    df_clean = df.copy()
    
    # Drop Loan_ID as it is just an identifier
    if 'Loan_ID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['Loan_ID'])
        
    # Remove duplicate records
    df_clean = df_clean.drop_duplicates()
    
    # If fill_values are not provided, we calculate them (usually for training data)
    if fill_values is None:
        fill_values = {
            'Gender': df_clean['Gender'].mode()[0],
            'Married': df_clean['Married'].mode()[0],
            'Dependents': df_clean['Dependents'].mode()[0],
            'Self_Employed': df_clean['Self_Employed'].mode()[0],
            'LoanAmount': df_clean['LoanAmount'].median(),
            'Loan_Amount_Term': df_clean['Loan_Amount_Term'].mode()[0],
            'Credit_History': df_clean['Credit_History'].mode()[0]
        }
        
    # Fill missing values
    for col, val in fill_values.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(val)
            
    # Correct data types
    # Dependents: replace '3+' with '3' and convert to int
    if 'Dependents' in df_clean.columns:
        df_clean['Dependents'] = df_clean['Dependents'].astype(str).str.replace('+', '', regex=False)
        df_clean['Dependents'] = df_clean['Dependents'].astype(int)
        
    # Credit_History and Loan_Amount_Term should be treated as categories/floats but filled
    if 'Credit_History' in df_clean.columns:
        df_clean['Credit_History'] = df_clean['Credit_History'].astype(float)
        
    if 'Loan_Amount_Term' in df_clean.columns:
        df_clean['Loan_Amount_Term'] = df_clean['Loan_Amount_Term'].astype(float)
        
    return df_clean, fill_values

def treat_outliers(df, cols=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']):
    """
    Applies log transformation to treat outliers and reduce skewness.
    - Uses log1p to handle 0 values in CoapplicantIncome.
    """
    df_treated = df.copy()
    for col in cols:
        if col in df_treated.columns:
            df_treated[f'{col}_log'] = np.log1p(df_treated[col])
    return df_treated

def encode_categorical(df, label_encoders=None, is_train=True):
    """
    Encodes categorical features.
    - Loan_Status: Label encoding (Y: 1, N: 0)
    - Other categorical features: One-hot encoding or Label encoding.
      We will use label encoding or manual mapping for binary features, and one-hot encoding for multiclass.
      Let's use a standard mapping to keep it clean.
    """
    df_encoded = df.copy()
    
    # Target variable encoding
    if 'Loan_Status' in df_encoded.columns:
        df_encoded['Loan_Status'] = df_encoded['Loan_Status'].map({'Y': 1, 'N': 0})
        
    # Categorical columns
    cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    
    if label_encoders is None:
        label_encoders = {}
        
    for col in cat_cols:
        if col in df_encoded.columns:
            if is_train:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                label_encoders[col] = le
            else:
                le = label_encoders.get(col)
                if le is not None:
                    # Handle unseen labels by mapping them to the first class or a default
                    df_encoded[col] = df_encoded[col].astype(str).map(
                        lambda s: s if s in le.classes_ else le.classes_[0]
                    )
                    df_encoded[col] = le.transform(df_encoded[col])
                    
    return df_encoded, label_encoders

def scale_features(df, scaler=None, num_cols=None, is_train=True):
    """
    Scales numerical features.
    """
    df_scaled = df.copy()
    if num_cols is None:
        num_cols = ['ApplicantIncome_log', 'CoapplicantIncome_log', 'LoanAmount_log', 
                    'Total_Income_log', 'EMI_Ratio', 'Loan_Income_Ratio']
        # Filter columns that are actually present
        num_cols = [c for c in num_cols if c in df_scaled.columns]
        
    if scaler is None and is_train:
        scaler = StandardScaler()
        df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])
    elif scaler is not None:
        df_scaled[num_cols] = scaler.transform(df_scaled[num_cols])
        
    return df_scaled, scaler
