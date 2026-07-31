import os
import pandas as pd
import joblib
from data_preprocessing import clean_data, treat_outliers, encode_categorical, scale_features
from feature_engineering import engineer_features
from model_training import train_and_compare

def run_pipeline():
    print("--- Starting Bank Loan Approval Pipeline ---")
    
    # 1. Load Raw Data
    raw_path = os.path.join("data", "raw", "train.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}. Run data_downloader.py first.")
        
    df_raw = pd.read_csv(raw_path)
    print(f"Loaded raw data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")
    
    # 2. Data Cleaning
    print("Cleaning data...")
    df_clean, fill_values = clean_data(df_raw)
    
    # 3. Feature Engineering
    print("Engineering features...")
    df_feat = engineer_features(df_clean)
    
    # 4. Outlier Treatment
    print("Treating outliers...")
    df_out = treat_outliers(df_feat)
    
    # Save the cleaned and engineered (but unscaled/unencoded) data for Dashboard and EDA
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    processed_path = os.path.join("data", "processed", "cleaned_loan_data.csv")
    df_out.to_csv(processed_path, index=False)
    print(f"Saved processed data to {processed_path}")
    
    # 5. Encoding Categorical
    print("Encoding categorical features...")
    df_encoded, label_encoders = encode_categorical(df_out, is_train=True)
    
    # 6. Scaling Features
    print("Scaling numerical features...")
    num_cols = [
        'ApplicantIncome_log', 'CoapplicantIncome_log', 'LoanAmount_log', 
        'Total_Income_log', 'EMI_Ratio', 'Loan_Income_Ratio'
    ]
    df_scaled, scaler = scale_features(df_encoded, num_cols=num_cols, is_train=True)
    
    # Save preprocessing assets
    os.makedirs("models", exist_ok=True)
    joblib.dump(fill_values, "models/fill_values.pkl")
    joblib.dump(label_encoders, "models/label_encoders.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("Saved preprocessing assets (fill_values, label_encoders, scaler).")
    
    # 7. Model Training & Comparison
    print("Training and comparing models...")
    comparison_df, best_model_name, trained_models = train_and_compare(df_scaled)
    
    print("\n--- Pipeline Completed Successfully! ---")
    print("Model Performance Summary:")
    print(comparison_df.to_string(index=False))

if __name__ == "__main__":
    run_pipeline()
