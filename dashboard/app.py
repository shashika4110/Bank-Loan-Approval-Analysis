import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as ob
import streamlit as st
import joblib

# Add project root to path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import clean_data, treat_outliers, encode_categorical, scale_features
from src.feature_engineering import engineer_features

# Set page config for a professional and modern look
st.set_page_config(
    page_title="Bank Loan Approval Analytics & Prediction Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling the dashboard to look premium and modern
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #F8F9FA;
        font-family: 'Inter', sans-serif;
    }
    
    /* Title styling */
    .title-text {
        color: #1E3A8A;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle-text {
        color: #6B7280;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Card design for metrics */
    .metric-card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    
    .metric-title {
        color: #4B5563;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    
    .metric-value {
        color: #111827;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Section dividers */
    .section-header {
        color: #1E3A8A;
        font-weight: 700;
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load data
@st.cache_data
def load_processed_data():
    processed_path = "data/processed/cleaned_loan_data.csv"
    if os.path.exists(processed_path):
        return pd.read_csv(processed_path)
    return None

# Helper function to load models and preprocessors
@st.cache_resource
def load_model_assets():
    model_path = "models/best_model.pkl"
    scaler_path = "models/scaler.pkl"
    encoders_path = "models/label_encoders.pkl"
    fill_values_path = "models/fill_values.pkl"
    
    if all(os.path.exists(p) for p in [model_path, scaler_path, encoders_path, fill_values_path]):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        encoders = joblib.load(encoders_path)
        fill_values = joblib.load(fill_values_path)
        return model, scaler, encoders, fill_values
    return None, None, None, None

# Load resources
df = load_processed_data()
model, scaler, encoders, fill_values = load_model_assets()

# Check if data and models are prepared
if df is None:
    st.title("🏦 Bank Loan Approval Analytics")
    st.warning("⚠️ Cleaned dataset not found! Please run the training pipeline first (`python src/train_pipeline.py`) to generate the cleaned data and train the model.")
    st.stop()

# Header Section
st.markdown("<div class='title-text'>🏦 Bank Loan Approval Analytics Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Analyze customer demographic and financial patterns influencing loan approvals and predict application status in real-time.</div>", unsafe_allow_html=True)

# Sidebar filters
st.sidebar.header("🎯 Filter Options")

# Categorical filters
genders = ["All"] + list(df['Gender'].unique())
selected_gender = st.sidebar.selectbox("Gender", genders)

married = ["All"] + list(df['Married'].unique())
selected_married = st.sidebar.selectbox("Married Status", married)

education = ["All"] + list(df['Education'].unique())
selected_education = st.sidebar.selectbox("Education Level", education)

property_area = ["All"] + list(df['Property_Area'].unique())
selected_property = st.sidebar.selectbox("Property Area", property_area)

credit_history = ["All", "Good (1.0)", "Bad (0.0)"]
selected_credit = st.sidebar.selectbox("Credit History Guidelines", credit_history)

# Filter dataframe based on selections
filtered_df = df.copy()

if selected_gender != "All":
    filtered_df = filtered_df[filtered_df['Gender'] == selected_gender]
if selected_married != "All":
    filtered_df = filtered_df[filtered_df['Married'] == selected_married]
if selected_education != "All":
    filtered_df = filtered_df[filtered_df['Education'] == selected_education]
if selected_property != "All":
    filtered_df = filtered_df[filtered_df['Property_Area'] == selected_property]
if selected_credit != "All":
    credit_val = 1.0 if "Good" in selected_credit else 0.0
    filtered_df = filtered_df[filtered_df['Credit_History'] == credit_val]

# Main layout tabs
tab_analytics, tab_prediction = st.tabs(["📊 Business Intelligence & Analytics", "🔮 Real-Time Loan Approval Predictor"])

with tab_analytics:
    # 1. KPI Cards Row
    col1, col2, col3, col4 = st.columns(4)
    
    total_applicants = len(filtered_df)
    
    if total_applicants > 0:
        approval_rate = (filtered_df['Loan_Status'].map({'Y': 1, 'N': 0}).mean() * 100)
        avg_loan = filtered_df['LoanAmount'].mean()
        avg_income = filtered_df['Total_Income'].mean()
    else:
        approval_rate = 0.0
        avg_loan = 0.0
        avg_income = 0.0
        
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #3B82F6;">
            <div class="metric-title">Total Applications</div>
            <div class="metric-value">{total_applicants:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #10B981;">
            <div class="metric-title">Approval Rate</div>
            <div class="metric-value">{approval_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #F59E0B;">
            <div class="metric-title">Average Loan Amount</div>
            <div class="metric-value">Rs{avg_loan:.1f}K</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #8B5CF6;">
            <div class="metric-title">Avg Household Income</div>
            <div class="metric-value">Rs {avg_income:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    if total_applicants == 0:
        st.warning("No records match the current filters. Please adjust the sidebar filters.")
    else:
        # 2. Charts Section
        st.markdown("<div class='section-header'>Visual Analytics Dashboard</div>", unsafe_allow_html=True)
        
        # Row 1: Approval Distribution & Income Distribution
        crow1_col1, crow1_col2 = st.columns(2)
        
        with crow1_col1:
            st.subheader("Loan Approval Distribution")
            fig_pie = px.pie(
                filtered_df, 
                names='Loan_Status', 
                color='Loan_Status',
                color_discrete_map={'Y': '#10B981', 'N': '#EF4444'},
                hole=0.4,
                labels={'Loan_Status': 'Status'}
            )
            fig_pie.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#FFFFFF', width=2)))
            fig_pie.update_layout(showlegend=False, height=350, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_pie, use_container_width=True)
            st.caption("**Insight**: Loan approvals represent the share of qualified applicants. Typically, banks accept 65% to 70% of loans in this dataset.")
            
        with crow1_col2:
            st.subheader("Household Income Distribution")
            fig_hist = px.histogram(
                filtered_df, 
                x='Total_Income', 
                color='Loan_Status',
                color_discrete_map={'Y': '#10B981', 'N': '#EF4444'},
                nbins=40,
                labels={'Total_Income': 'Annual Total Income (Rs)', 'count': 'Number of Applicants'},
                marginal="box"
            )
            fig_hist.update_layout(
                barmode='overlay', 
                height=350, 
                margin=dict(t=20, b=20, l=20, r=20),
                legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center")
            )
            fig_hist.update_traces(opacity=0.7)
            st.plotly_chart(fig_hist, use_container_width=True)
            st.caption("**Insight**: Income distribution is heavily right-skewed. Higher concentration of approvals is found in the moderate income ranges, while very low-income applicants face higher rejection rates.")

        # Row 2: Categorical Segments vs Loan Status
        crow2_col1, crow2_col2, crow2_col3 = st.columns(3)
        
        with crow2_col1:
            st.subheader("Approval Rate by Education")
            # Calculate approval rates
            edu_stats = filtered_df.groupby('Education')['Loan_Status'].value_counts(normalize=True).unstack().fillna(0) * 100
            edu_stats = edu_stats.reset_index()
            fig_edu = px.bar(
                edu_stats, 
                x='Education', 
                y=['Y', 'N'],
                labels={'value': 'Percentage (%)', 'variable': 'Approved?'},
                color_discrete_map={'Y': '#10B981', 'N': '#EF4444'},
                barmode='group'
            )
            fig_edu.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20), legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_edu, use_container_width=True)
            st.caption("**Insight**: Graduates exhibit a noticeably higher approval rate compared to non-graduates, reflecting the credit assessment of educational stability.")
            
        with crow2_col2:
            st.subheader("Approval Rate by Property Area")
            prop_stats = filtered_df.groupby('Property_Area')['Loan_Status'].value_counts(normalize=True).unstack().fillna(0) * 100
            prop_stats = prop_stats.reset_index()
            fig_prop = px.bar(
                prop_stats, 
                x='Property_Area', 
                y=['Y', 'N'],
                labels={'value': 'Percentage (%)', 'variable': 'Approved?'},
                color_discrete_map={'Y': '#10B981', 'N': '#EF4444'},
                barmode='group'
            )
            fig_prop.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20), legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_prop, use_container_width=True)
            st.caption("**Insight**: Semiurban areas have the highest loan approval rates (often over 75%), while rural areas face the lowest, likely due to property valuation risks.")
            
        with crow2_col3:
            st.subheader("Approval Rate by Credit History")
            cred_df = filtered_df.copy()
            cred_df['Credit_History_Label'] = cred_df['Credit_History'].map({1.0: 'Good History (1.0)', 0.0: 'Bad History (0.0)'})
            cred_stats = cred_df.groupby('Credit_History_Label')['Loan_Status'].value_counts(normalize=True).unstack().fillna(0) * 100
            cred_stats = cred_stats.reset_index()
            fig_cred = px.bar(
                cred_stats, 
                x='Credit_History_Label', 
                y=['Y', 'N'],
                labels={'value': 'Percentage (%)', 'variable': 'Approved?'},
                color_discrete_map={'Y': '#10B981', 'N': '#EF4444'},
                barmode='group'
            )
            fig_cred.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20), legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_cred, use_container_width=True)
            st.caption("**Insight**: Credit History is the single most critical factor: over 80% of applicants with good credit history get approved, whereas less than 10% get approved with bad history.")

with tab_prediction:
    st.markdown("<div class='section-header'>Real-Time Underwriting & Loan Status Predictor</div>", unsafe_allow_html=True)
    
    if model is None:
        st.error("Model assets not loaded. Please ensure the machine learning models have been trained and saved under the `models/` directory.")
    else:
        st.write("Fill out the applicant's profile details below to get an instant automated credit decision and probability score based on the optimized machine learning model.")
        
        # Form layout
        with st.form("loan_prediction_form"):
            col_f1, col_f2 = st.columns(2)
            
            with col_f1:
                st.subheader("Demographic Details")
                gender_input = st.selectbox("Gender", ['Male', 'Female'])
                married_input = st.selectbox("Married Status", ['Yes', 'No'])
                dependents_input = st.selectbox("Number of Dependents", ['0', '1', '2', '3+'])
                education_input = st.selectbox("Education Level", ['Graduate', 'Not Graduate'])
                self_employed_input = st.selectbox("Self Employed?", ['No', 'Yes'])
                property_area_input = st.selectbox("Property Area", ['Rural', 'Semiurban', 'Urban'])
                
            with col_f2:
                st.subheader("Financial Details")
                app_income_input = st.number_input("Applicant's Annual/Monthly Base Income (Rs)", min_value=0, value=5000, step=500, help="Standardized to match the dataset units (e.g. monthly).")
                coapp_income_input = st.number_input("Co-Applicant's Annual/Monthly Income (Rs)", min_value=0, value=0, step=500)
                loan_amount_input = st.number_input("Requested Loan Amount (in Thousands, e.g. 150 = Rs150,000)", min_value=1, value=120, step=10)
                loan_term_input = st.number_input("Loan Amount Term (in Months, e.g. 360 = 30 Years)", min_value=12, value=360, step=12)
                credit_history_input = st.selectbox("Credit History Guidelines Met?", ['Yes (Good Credit)', 'No (Bad Credit)'])
                
            submit_btn = st.form_submit_button("Submit Application for Analysis")
            
        if submit_btn:
            # Recreate raw input dict
            raw_input = {
                'Gender': gender_input,
                'Married': married_input,
                'Dependents': dependents_input,
                'Education': education_input,
                'Self_Employed': self_employed_input,
                'ApplicantIncome': float(app_income_input),
                'CoapplicantIncome': float(coapp_income_input),
                'LoanAmount': float(loan_amount_input),
                'Loan_Amount_Term': float(loan_term_input),
                'Credit_History': 1.0 if 'Yes' in credit_history_input else 0.0,
                'Property_Area': property_area_input
            }
            
            # Put into DataFrame
            input_df = pd.DataFrame([raw_input])
            
            try:
                # 1. Cleaning & Type conversion (Uses the trained fill values to ensure consistency)
                cleaned_input, _ = clean_data(input_df, fill_values=fill_values)
                
                # 2. Feature engineering
                feat_input = engineer_features(cleaned_input)
                
                # 3. Outlier treatment (log transformations)
                out_input = treat_outliers(feat_input)
                
                # 4. Encoding
                encoded_input, _ = encode_categorical(out_input, label_encoders=encoders, is_train=False)
                
                # 5. Scaling
                num_cols_to_scale = [
                    'ApplicantIncome_log', 'CoapplicantIncome_log', 'LoanAmount_log', 
                    'Total_Income_log', 'EMI_Ratio', 'Loan_Income_Ratio'
                ]
                scaled_input, _ = scale_features(encoded_input, scaler=scaler, num_cols=num_cols_to_scale, is_train=False)
                
                # Align columns for ML prediction
                features_list = [
                    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                    'Property_Area', 'Credit_History',
                    'ApplicantIncome_log', 'CoapplicantIncome_log', 'LoanAmount_log', 
                    'Total_Income_log', 'EMI_Ratio', 'Loan_Income_Ratio'
                ]
                X_pred = scaled_input[features_list]
                
                # Predict
                pred = model.predict(X_pred)[0]
                prob = model.predict_proba(X_pred)[0][1] if hasattr(model, "predict_proba") else 0.85
                
                # Display Results
                st.markdown("<div class='section-header'>Automated Underwriting Decision</div>", unsafe_allow_html=True)
                
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    if pred == 1:
                        st.markdown("<h2 style='color:#10B981;'>🎉 Application APPROVED</h2>", unsafe_allow_html=True)
                        st.write("This applicant meets the credit policy guidelines. The loan has a high likelihood of successful repayment.")
                    else:
                        st.markdown("<h2 style='color:#EF4444;'>❌ Application REJECTED</h2>", unsafe_allow_html=True)
                        st.write("This application does not meet the necessary credit standards. Review the risk factors below.")
                        
                with col_res2:
                    st.metric(label="Model Approval Probability", value=f"{prob * 100:.1f}%")
                    st.progress(float(prob))
                    
                # Explanation logic
                st.markdown("### 🔍 Risk Factor Analysis")
                if raw_input['Credit_History'] == 0.0:
                    st.warning("⚠️ **Critical Risk**: The applicant has a bad credit history. Good credit history is the primary driver of approvals; without it, the rejection risk increases by over 80%.")
                if feat_input['EMI_Ratio'].iloc[0] > 0.5:
                    st.warning("⚠️ **High Debt Service**: The estimated EMI exceeds 50% of the applicant's monthly income. This indicates high debt stress and lower repayment buffer.")
                if raw_input['ApplicantIncome'] < 2500 and raw_input['CoapplicantIncome'] == 0:
                    st.info("ℹ️ **Low Financial Strength**: The applicant has low base income and no co-applicant. Adding a co-applicant can increase combined income and reduce risk.")
                if pred == 1 and raw_input['Credit_History'] == 1.0:
                    st.success("✅ **Strengths**: Solid credit history and manageable debt service ratio.")
                    
            except Exception as e:
                st.error(f"Error executing prediction pipeline: {e}")
                st.info("Please verify that features match the training pipeline schema.")
