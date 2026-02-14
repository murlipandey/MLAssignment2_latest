import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Classification Models",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🤖 ML Classification Models")
st.markdown("**Multi-Model Classification System** - Compare 6 different ML models")

# Sidebar
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["Home", "Model Evaluation", "Dataset Upload & Prediction"]
)

# Load model results
@st.cache_data
def load_model_results():
    if os.path.exists('model_results.csv'):
        return pd.read_csv('model_results.csv', index_col=0)
    return None

# Load trained models
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'Logistic Regression': 'models/logistic_regression.pkl',
        'Decision Tree': 'models/decision_tree.pkl',
        'KNN': 'models/knn.pkl',
        'Naive Bayes': 'models/naive_bayes.pkl',
        'Random Forest': 'models/random_forest.pkl',
        'XGBoost': 'models/xgboost.pkl'
    }
    
    for model_name, model_path in model_files.items():
        if os.path.exists(model_path):
            models[model_name] = joblib.load(model_path)
    
    return models

# Load scaler and encoder
@st.cache_resource
def load_preprocessing_tools():
    scaler = None
    encoder = None
    
    if os.path.exists('models/scaler.pkl'):
        scaler = joblib.load('models/scaler.pkl')
    
    if os.path.exists('models/label_encoder.pkl'):
        encoder = joblib.load('models/label_encoder.pkl')
    
    return scaler, encoder

# PAGE 1: HOME
if page == "Home":
    st.header("📊 Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Problem Statement")
        st.markdown("""
        This project implements and compares **6 different classification models** 
        to classify animals (Cats, Dogs, Horses) using machine learning algorithms.
        """)
    
    with col2:
        st.subheader("🎯 Objectives")
        st.markdown("""
        - Train 6 ML classification models
        - Evaluate using 6 different metrics
        - Compare model performance
        - Deploy as interactive web application
        """)
    
    st.divider()
    
    st.subheader("📚 Models Implemented")
    
    models_info = {
        "Logistic Regression": "Linear model for classification",
        "Decision Tree": "Tree-based model for interpretability",
        "K-Nearest Neighbor (KNN)": "Instance-based learning algorithm",
        "Naive Bayes": "Probabilistic classifier based on Bayes theorem",
        "Random Forest": "Ensemble method using multiple decision trees",
        "XGBoost": "Gradient boosting ensemble method"
    }
    
    cols = st.columns(3)
    for idx, (model_name, description) in enumerate(models_info.items()):
        with cols[idx % 3]:
            st.info(f"**{model_name}**\n\n{description}")
    
    st.divider()
    
    st.subheader("📊 Evaluation Metrics")
    st.markdown("""
    - **Accuracy**: Percentage of correct predictions
    - **AUC Score**: Area under the ROC curve
    - **Precision**: Correctly predicted positive cases / Total predicted positive cases
    - **Recall**: Correctly predicted positive cases / Total actual positive cases
    - **F1 Score**: Harmonic mean of precision and recall
    - **MCC (Matthews Correlation Coefficient)**: Correlation between predicted and actual values
    """)

# PAGE 2: MODEL EVALUATION
elif page == "Model Evaluation":
    st.header("📊 Model Evaluation Metrics")
    
    results_df = load_model_results()
    
    if results_df is not None:
        st.subheader("Metrics Comparison Table")
        
        # Display metrics table
        st.dataframe(
            results_df.round(4),
            use_container_width=True,
            column_config={
                "Accuracy": st.column_config.NumberColumn(format="%.4f"),
                "AUC": st.column_config.NumberColumn(format="%.4f"),
                "Precision": st.column_config.NumberColumn(format="%.4f"),
                "Recall": st.column_config.NumberColumn(format="%.4f"),
                "F1": st.column_config.NumberColumn(format="%.4f"),
                "MCC": st.column_config.NumberColumn(format="%.4f"),
            }
        )
        
        # Display individual model metrics
        st.divider()
        st.subheader("🎯 Individual Model Performance")
        
        selected_model = st.selectbox(
            "Select a model to view detailed metrics:",
            results_df.index.tolist()
        )
        
        if selected_model:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Accuracy", f"{results_df.loc[selected_model, 'Accuracy']:.4f}")
                st.metric("Precision", f"{results_df.loc[selected_model, 'Precision']:.4f}")
            
            with col2:
                st.metric("AUC", f"{results_df.loc[selected_model, 'AUC']:.4f}")
                st.metric("Recall", f"{results_df.loc[selected_model, 'Recall']:.4f}")
            
            with col3:
                st.metric("F1 Score", f"{results_df.loc[selected_model, 'F1']:.4f}")
                st.metric("MCC", f"{results_df.loc[selected_model, 'MCC']:.4f}")
        
        # Visualization
        st.divider()
        st.subheader("📈 Metrics Visualization")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Comparison Across Metrics', fontsize=16, fontweight='bold')
        
        metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            axes[idx].barh(results_df.index, results_df[metric], color='steelblue')
            axes[idx].set_xlabel(metric)
            axes[idx].set_xlim([0, 1])
            axes[idx].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    else:
        st.warning("⚠️ Model results not found. Please train the models first.")
        st.info("Run `python train_models.py` with your dataset to generate results.")

# PAGE 3: DATASET UPLOAD & PREDICTION
elif page == "Dataset Upload & Prediction":
    st.header("🔮 Make Predictions")
    
    models = load_models()
    scaler, encoder = load_preprocessing_tools()
    
    if len(models) == 0:
        st.error("❌ No trained models found. Please train the models first.")
    else:
        st.subheader("📤 Upload Your Dataset")
        
        uploaded_file = st.file_uploader(
            "Upload a CSV file for prediction",
            type=['csv'],
            help="Upload test data in CSV format"
        )
        
        if uploaded_file is not None:
            # Read uploaded file
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✓ File uploaded successfully! Shape: {df.shape}")
            
            st.subheader("📋 Dataset Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Model selection
            st.divider()
            st.subheader("🤖 Select Model for Prediction")
            
            selected_model = st.selectbox(
                "Choose a model:",
                list(models.keys())
            )
            
            if st.button("🚀 Make Predictions", key="predict_button"):
                try:
                    # Preprocess data
                    if scaler is not None:
                        X_scaled = scaler.transform(df)
                    else:
                        X_scaled = df.values
                    
                    # Make predictions
                    model = models[selected_model]
                    predictions = model.predict(X_scaled)
                    
                    # Decode predictions if encoder exists
                    if encoder is not None:
                        predictions_labels = encoder.inverse_transform(predictions)
                    else:
                        predictions_labels = predictions
                    
                    # Display results
                    st.success("✓ Predictions completed!")
                    
                    st.subheader("📊 Prediction Results")
                    
                    results_df = pd.DataFrame({
                        'Prediction': predictions_labels
                    })
                    
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
        
        else:
            st.info("👆 Upload a CSV file to get started")

# Footer
st.divider()
st.markdown("""
---
**ML Classification Assignment** | Cats, Dogs & Horses Classification
""")
