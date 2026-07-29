import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib

def evaluate_model(model, X_test, y_test, name):
    """
    Evaluates a model and returns a dictionary of metrics along with ROC and Confusion Matrix data.
    """
    y_pred = model.predict(X_test)
    
    # Check if model has predict_proba
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
        # normalize to [0,1] range for ROC-AUC
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
    else:
        y_prob = y_pred
        
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    metrics = {
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC-AUC': roc_auc
    }
    
    return metrics, cm, fpr, tpr, y_prob

def train_and_compare(df, target_col='Loan_Status', test_size=0.2, random_state=42):
    """
    Split data, train models, compare performance, generate evaluation plots, and save the best model.
    """
    # Define features to use
    features = [
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
        'Property_Area', 'Credit_History',
        'ApplicantIncome_log', 'CoapplicantIncome_log', 'LoanAmount_log', 
        'Total_Income_log', 'EMI_Ratio', 'Loan_Income_Ratio'
    ]
    
    X = df[features]
    y = df[target_col]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_state),
        'Support Vector Machine': SVC(probability=True, random_state=random_state),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=3, random_state=random_state)
    }
    
    results = []
    confusion_matrices = {}
    roc_curves = {}
    trained_models = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        metrics, cm, fpr, tpr, y_prob = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        confusion_matrices[name] = cm
        roc_curves[name] = (fpr, tpr, metrics['ROC-AUC'])
        
    comparison_df = pd.DataFrame(results)
    
    # Create directories for models and images if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    
    # Save the comparison table
    comparison_df.to_csv("models/model_comparison.csv", index=False)
    
    # Select Best Model based on F1 Score or Accuracy (let's use F1 Score as it is balanced)
    best_model_name = comparison_df.sort_values(by='F1 Score', ascending=False).iloc[0]['Model']
    best_model = trained_models[best_model_name]
    print(f"\nBest Performing Model (based on F1 Score): {best_model_name}")
    
    # Save the best model
    joblib.dump(best_model, "models/best_model.pkl")
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    for name, (fpr, tpr, auc_val) in roc_curves.items():
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_val:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("images/roc_curve.png", dpi=300)
    plt.close()
    
    # Plot Confusion Matrices side-by-side or in grid
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    axes = axes.ravel()
    
    for idx, (name, cm) in enumerate(confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                    xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'])
        axes[idx].set_title(f'{name} Confusion Matrix', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_ylabel('True Label')
        
    # Remove empty subplot
    if len(models) < len(axes):
        fig.delaxes(axes[len(models)])
        
    plt.tight_layout()
    plt.savefig("images/confusion_matrices.png", dpi=300)
    plt.close()
    
    # Feature Importance for the best model if available
    plot_feature_importance(best_model, best_model_name, features)
    
    return comparison_df, best_model_name, trained_models

def plot_feature_importance(model, model_name, feature_names):
    """
    Plots feature importance for Tree-based models or coefficients for Logistic Regression.
    """
    plt.figure(figsize=(10, 6))
    importance = []
    
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        title = f'Feature Importance - {model_name}'
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
        title = f'Feature Importance (Absolute Coefficients) - {model_name}'
    else:
        print(f"Model {model_name} does not support direct feature importance calculation.")
        return
        
    feat_imp = pd.Series(importance, index=feature_names).sort_values(ascending=True)
    
    # Custom color palette matching a professional look
    colors = sns.color_palette("viridis", len(feat_imp))
    
    feat_imp.plot(kind='barh', color=colors, edgecolor='black', alpha=0.8)
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig("images/feature_importance.png", dpi=300)
    plt.close()
