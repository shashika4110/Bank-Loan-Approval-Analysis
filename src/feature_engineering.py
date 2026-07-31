import numpy as np
import pandas as pd

def engineer_features(df):
    """
    Creates engineered features:
    - Total_Income: ApplicantIncome + CoapplicantIncome
    - EMI: (LoanAmount * 1000) / Loan_Amount_Term (estimated monthly payment)
    - EMI_Ratio: EMI / (Total_Income / 12) (DTI approximation)
    - Loan_Income_Ratio: (LoanAmount * 1000) / Total_Income
    - Income_Category: Low, Medium, High based on Total_Income
    """
    df_feat = df.copy()
    
    # 1. Total Income (Combine applicant and coapplicant income)
    df_feat['Total_Income'] = df_feat['ApplicantIncome'] + df_feat['CoapplicantIncome']
    df_feat['Total_Income_log'] = np.log1p(df_feat['Total_Income'])
    
    # 2. Monthly EMI Approximation (LoanAmount is in thousands, Term is in months)
    # Handle division by zero or NaN values in Loan_Amount_Term
    df_feat['EMI'] = np.where(
        df_feat['Loan_Amount_Term'] > 0,
        (df_feat['LoanAmount'] * 1000) / df_feat['Loan_Amount_Term'],
        0
    )
    
    # 3. EMI Ratio (Monthly EMI / Monthly Income)
    df_feat['EMI_Ratio'] = np.where(
        df_feat['Total_Income'] > 0,
        df_feat['EMI'] / (df_feat['Total_Income'] / 12),
        0
    )
    
    # 4. Loan to Income Ratio (Total Loan Amount / Annual Income)
    df_feat['Loan_Income_Ratio'] = np.where(
        df_feat['Total_Income'] > 0,
        (df_feat['LoanAmount'] * 1000) / df_feat['Total_Income'],
        0
    )
    
    # 5. Income Category (Segment based on Total Income)
    # Define bounds: Low (< 4000), Medium (4000 to 8000), High (> 8000)
    def categorize_income(income):
        if income < 4000:
            return 'Low'
        elif income <= 8000:
            return 'Medium'
        else:
            return 'High'
            
    df_feat['Income_Category'] = df_feat['Total_Income'].apply(categorize_income)
    
    return df_feat
