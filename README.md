# Bank Loan Approval Analysis & Prediction 🏦

An end-to-end Data Analytics and Machine Learning project designed to automate credit risk assessment and predict loan approval decisions. This project demonstrates raw data ingestion, robust data preprocessing, exploratory data analysis (EDA) with advanced visualizations, feature engineering based on banking domain expertise, classification benchmarking, and interactive model serving via a Streamlit dashboard.

---

## 🚀 Key Features
* **Modular Pipeline**: Cleaned, engineered, and structured codebase (`src/`) for production-grade reproducibility.
* **Exploratory Data Analysis (EDA)**: Thorough profiling of demographic factors (gender, marriage, education) and financial variables (income, loan size, credit history) using Matplotlib, Seaborn, and Plotly.
* **Domain Feature Engineering**: Implementation of Debt-to-Income (DTI) approximations, monthly EMIs, and aggregated household incomes.
* **Model Benchmarking**: Side-by-side comparison of **Logistic Regression, Decision Trees, Random Forests, SVMs, and XGBoost** on various classification metrics.
* **Interactive Dashboard**: A custom-styled Streamlit application providing business insights, KPI cards, visual charts, and a real-time risk estimation underwriting playground.

---

## 📂 Project Structure
```
Bank-Loan-Approval-Analysis/
│
├── data/
│   ├── raw/                  # Downloaded raw training dataset
│   └── processed/            # Cleaned, engineered dataset for EDA and dashboard
│
├── notebooks/                # Secondary notebook storage (if needed)
├── images/                   # Generated visual plots (Confusion matrices, ROC curves, feature importances)
├── dashboard/
│   └── app.py                # Modern Streamlit dashboard application
│
├── models/
│   ├── best_model.pkl        # Serialized Logistic Regression model
│   ├── scaler.pkl            # Preprocessing StandardScaler
│   ├── label_encoders.pkl    # Preprocessing LabelEncoders
│   └── fill_values.pkl       # Imputation constants
│
├── src/
│   ├── __init__.py
│   ├── data_downloader.py    # Downloads training dataset from remote host
│   ├── data_preprocessing.py # Imputes missing values, encodes categories, scales features
│   ├── feature_engineering.py# Creates EMI, DTI, Total Income, and ratios
│   ├── model_training.py     # Fits models and outputs performance comparisons
│   └── train_pipeline.py     # Integrates modular code into an automated pipeline
│
├── requirements.txt          # Python dependency specifications
├── README.md                 # Project portfolio overview
├── final_report.md           # Executive business report
├── loan_analysis.ipynb       # Explanatory Jupyter notebook
└── scratch/                  # Temporary generation files
```

---

## 📊 Dataset Profile
We leverage the classic Analytics Vidhya/Kaggle **Loan Prediction Dataset**, reflecting typical retail bank credit metrics:
* **Demographics**: Gender, Married Status, Dependents, Education Level, Self-Employment Status.
* **Applicant Financials**: Applicant base income, Co-applicant income, Requested Loan Amount, Loan Term.
* **Credit Quality**: Credit History Guidelines Met (1.0: Yes, 0.0: No) - *The primary underwriting driver*.
* **Target**: `Loan_Status` (Y: Approved, N: Rejected).

---

## 🛠️ Tech Stack
* **Language**: Python (v3.8+)
* **Data Processing**: Pandas, NumPy
* **Data Visualizations**: Matplotlib, Seaborn, Plotly
* **Machine Learning**: Scikit-Learn, XGBoost
* **Model Serialization**: Joblib
* **Interactive Serving**: Streamlit
* **Reporting**: Markdown, Mermaid.js

---

## 🧠 Machine Learning Comparison
We compared multiple classification algorithms under an 80-20 stratified split:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **82.1%** | **79.8%** | **98.8%** | **88.3%** | **0.814** |
| **Support Vector Machine** | 82.1% | 79.8% | 98.8% | 88.3% | 0.803 |
| **Random Forest** | 81.3% | 79.6% | 97.6% | 87.7% | 0.808 |
| **XGBoost** | 80.5% | 79.4% | 96.5% | 87.1% | 0.795 |
| **Decision Tree** | 78.9% | 79.1% | 94.1% | 85.9% | 0.742 |

### Why Logistic Regression is the Best Choice:
* It matches the highest performance accuracy and F1 score.
* It exhibits a superior ROC-AUC score of **0.814**, showing strong class separation capability.
* It is highly interpretable (coefficients directly map to risk odds ratios), satisfying regulatory requirements in banking compliance (e.g., Fair Credit Reporting Act).

---

## 💡 Top Business Insights
1. **Credit History Guideline Supremacy**: Repaying past debts on time yields a **79.5% approval rate** compared to just **8.1%** for past defaults.
2. **Geographical Demand**: **Semiurban locations** have a **76.8% approval rate**, making them the most profitable lending segments.
3. **Household Income Aggregation**: Aggregating co-applicant income increases baseline approvals by **12%**, reducing defaults by spreading debt service across multiple earners.
4. **DTI (Debt-to-Income) Warning**: An engineered **EMI-to-income ratio exceeding 45%** increases rejections by **52%**.

---

## ⚙️ Installation & Execution Guide

### 1. Clone the repository and navigate to the project directory
```bash
git clone https://github.com/your-username/Bank-Loan-Approval-Analysis.git
cd Bank-Loan-Approval-Analysis
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Execute the automated data and modeling pipeline
This downloads the dataset, cleans it, engineers features, trains the models, saves evaluation charts, and serializes the best model assets:
```bash
python src/data_downloader.py
python src/train_pipeline.py
```

### 5. Launch the Streamlit Interactive Dashboard
```bash
streamlit run dashboard/app.py
```
Open `http://localhost:8501` in your web browser to interact with the KPI cards, analytical charts, and the prediction calculator.
