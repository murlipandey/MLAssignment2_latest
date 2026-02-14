# 🤖 ML Classification Project: Cats, Dogs & Horses

## 📋 Project Overview

This project demonstrates a complete machine learning pipeline for **multi-class animal classification** (Cats, Dogs, Horses). It includes:
- Synthetic dataset generation (1,200 samples, 15 features)
- Implementation of 6 different classification algorithms
- Comprehensive model evaluation using 6 metrics
- Interactive Streamlit web application for predictions
- Ready for deployment on Streamlit Community Cloud

### 🎯 Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to accurately classify animal data into three categories: **Cats, Dogs, and Horses**. This is a multi-class classification problem that demonstrates the implementation and evaluation of different ML algorithms.

## 📊 Dataset Description

**Dataset Name**: Animal Classification Dataset (Cats, Dogs, Horses)  
**Source**: Synthetically Generated  
**Number of Instances**: 1,200  
**Number of Features**: 15  
**Classes**: 3 (Cat, Dog, Horse)  
**Feature Types**: Numeric (continuous)

### Dataset Overview
- **Total Samples**: 1,200 (balanced - 400 each class)
- **Training Set**: 960 samples (80%)
- **Test Set**: 240 samples (20%)
- **Feature Scaling**: StandardScaler applied to all features
- **Data Preprocessing**: No missing values, all numeric features

### Features (15 characteristics):
1. head_size - Size of the animal's head
2. body_length - Total body length
3. ear_size - Size of ears
4. ear_shape - Shape characteristic of ears
5. tail_length - Length of tail
6. tail_thickness - Thickness of tail
7. leg_length - Average leg length
8. leg_count - Number of legs
9. claw_sharpness - Sharpness of claws/nails
10. bite_force - Estimated bite force
11. speed_capability - Maximum running speed
12. teeth_count - Number of teeth
13. coat_density - Hair/coat density
14. whisker_presence - Presence of whiskers
15. hoof_presence - Presence of hooves

### Class Distribution
- **Cat**: 400 samples
- **Dog**: 400 samples
- **Horse**: 400 samples

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (for version control)
- ~500MB free disk space

### Step 1: Clone or Download the Project

```bash
# Clone from GitHub (if using Git)
git clone <your-github-repo-url>
cd MLAssignment2_latest

# OR navigate to project folder
cd c:\Users\Harsh Pandey\project\MLAssignment2_latest
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**Key Dependencies:**
- pandas, numpy - Data manipulation
- scikit-learn - ML algorithms
- xgboost - Gradient boosting
- matplotlib, seaborn - Visualization
- streamlit - Web application

### Step 4: Verify Installation

```bash
python -c "import pandas, sklearn, xgboost, streamlit; print('✓ All packages installed')"
```

---

## 🚀 How to Execute the Project

### Quick Start (Recommended)

The fastest way to see the project in action:

```bash
# Simply run the Streamlit app
streamlit run app.py
```

Then open your browser to: **http://localhost:8501**

### Detailed Execution Steps

#### Option A: Generate New Dataset

To create a fresh dataset with different parameters:

```bash
python create_dataset.py
```

**What happens:**
- Generates 1,200 synthetic animal samples
- Creates 15 feature columns
- Balances classes (400 each)
- Saves to `animal_classification_dataset.csv`
- Displays statistics and sample data

**Output:** `animal_classification_dataset.csv`

#### Option B: Train All Models

To retrain models from scratch:

```bash
python train_models.py
```

**What happens:**
1. Loads the dataset
2. Preprocesses data (scaling, encoding, splitting)
3. Trains all 6 ML models:
   - Logistic Regression
   - Decision Tree
   - K-Nearest Neighbor
   - Naive Bayes
   - Random Forest
   - XGBoost
4. Evaluates using 6 metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
5. Saves trained models as .pkl files
6. Generates results summary

**Output Files:**
```
models/
├── logistic_regression.pkl
├── decision_tree.pkl
├── knn.pkl
├── naive_bayes.pkl
├── random_forest.pkl
├── xgboost.pkl
├── scaler.pkl
└── label_encoder.pkl

model_results.csv (metrics summary)
```

**Expected Runtime:** ~30-60 seconds

#### Option C: Run Streamlit Application

The interactive web interface:

```bash
streamlit run app.py
```

**Output:**
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

**Access the app:**
- Open browser
- Go to `http://localhost:8501`
- Use sidebar to navigate between pages

---

## 📱 Streamlit Application Guide

### Page 1: Home
**Purpose:** Project overview and information

**What to do:**
1. Read project objectives
2. Learn about the 6 models
3. Understand evaluation metrics
4. Review problem statement

**Key Sections:**
- Problem Statement
- Models Implemented
- Evaluation Metrics

### Page 2: Model Evaluation
**Purpose:** Compare model performance

**What to do:**
1. View metrics comparison table (all 6 models)
2. Select a specific model from dropdown
3. See detailed metrics in card format
4. View visual comparisons in charts

**Available Metrics:**
- Accuracy, AUC, Precision, Recall, F1 Score, MCC

**Visualizations:**
- Bar charts comparing metrics
- Individual model performance cards

### Page 3: Dataset Upload & Prediction
**Purpose:** Make predictions on new data

**Step-by-step guide:**

1. **Prepare your CSV file:**
   - Must have exactly 15 columns
   - Column names must match training data:
   ```
   head_size, body_length, ear_size, ear_shape, tail_length, 
   tail_thickness, leg_length, leg_count, claw_sharpness, 
   bite_force, speed_capability, teeth_count, coat_density, 
   whisker_presence, hoof_presence
   ```
   - Example:
   ```
   head_size,body_length,ear_size,ear_shape,...
   7.5,35.0,8.0,8.0,...
   25.0,180.0,12.0,6.0,...
   ```

2. **Upload the file:**
   - Click "Browse files" button
   - Select your CSV file
   - Wait for success message

3. **Select a model:**
   - Choose from dropdown:
     - Logistic Regression
     - Decision Tree
     - KNN
     - Naive Bayes
     - Random Forest
     - XGBoost

4. **Make predictions:**
   - Click "🚀 Make Predictions" button
   - Wait for results
   - View predictions table

5. **Download results:**
   - Click "📥 Download Results (CSV)"
   - File saved to your downloads

**Example CSV file to test:**

Create a file `test_data.csv`:
```
head_size,body_length,ear_size,ear_shape,tail_length,tail_thickness,leg_length,leg_count,claw_sharpness,bite_force,speed_capability,teeth_count,coat_density,whisker_presence,hoof_presence
7.5,35.0,8.0,8.0,25.0,2.0,15.0,4,8.5,150,48,30,6.0,9,1
25.0,180.0,12.0,6.0,90.0,8.0,90.0,4,3.0,400,88,44,6.5,2,9
10.0,50.0,9.0,5.0,30.0,3.0,20.0,4,5.0,200,40,42,7.0,3,1
```

---

## 🤖 Models Used

This project implements and evaluates the following 6 machine learning classification models:

### 1. Logistic Regression
- **Description**: Linear model for classification based on probability
- **Hyperparameters**: random_state=42, max_iter=1000
- **Advantages**: Fast, interpretable, works well with linear relationships
- **Disadvantages**: Assumes linear decision boundaries

### 2. Decision Tree Classifier
- **Description**: Tree-based model that makes decisions by splitting on features
- **Hyperparameters**: random_state=42
- **Advantages**: Highly interpretable, handles non-linear relationships
- **Disadvantages**: Prone to overfitting

### 3. K-Nearest Neighbor (KNN)
- **Description**: Instance-based learning that classifies based on nearest neighbors
- **Hyperparameters**: n_neighbors=5
- **Advantages**: Simple, no training phase, effective for small datasets
- **Disadvantages**: Slow prediction time, sensitive to feature scaling

### 4. Naive Bayes Classifier
- **Description**: Probabilistic classifier based on Bayes' theorem
- **Type**: Gaussian Naive Bayes (for continuous features)
- **Advantages**: Fast, works well with high-dimensional data
- **Disadvantages**: Assumes feature independence

### 5. Random Forest Classifier (Ensemble)
- **Description**: Ensemble method using multiple decision trees
- **Hyperparameters**: n_estimators=100, random_state=42
- **Advantages**: Reduces overfitting, handles non-linear data, feature importance
- **Disadvantages**: Less interpretable than single tree

### 6. XGBoost Classifier (Ensemble)
- **Description**: Gradient boosting ensemble method with optimized learning
- **Hyperparameters**: n_estimators=100, random_state=42
- **Advantages**: High performance, handles missing values, feature importance
- **Disadvantages**: Requires tuning, more computational resources

---

## 📈 Model Performance Comparison

### Evaluation Metrics Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| KNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Naive Bayes | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost (Ensemble) | 0.9958 | 1.0000 | 0.9959 | 0.9958 | 0.9958 | 0.9938 |

### Metric Definitions

- **Accuracy**: Percentage of correct predictions out of total predictions
  - Formula: (TP + TN) / (TP + TN + FP + FN)

- **AUC (Area Under Curve)**: Area under ROC curve, measures discrimination ability
  - Range: 0 to 1 (higher is better)

- **Precision**: Of predicted positives, how many are actually positive
  - Formula: TP / (TP + FP)

- **Recall**: Of actual positives, how many were predicted correctly
  - Formula: TP / (TP + FN)

- **F1 Score**: Harmonic mean of Precision and Recall
  - Formula: 2 * (Precision * Recall) / (Precision + Recall)

- **MCC (Matthews Correlation Coefficient)**: Correlation between predicted and actual values
  - Range: -1 to 1 (higher is better)

---

## 🔍 Model Performance Observations

### 1. Logistic Regression
**Observations**:
- Achieved perfect 100% accuracy on the test set
- Outstanding performance across all metrics
- Linear decision boundaries effectively separate the three animal classes
- Fast training and inference time
- Demonstrates that the dataset has well-separated linear patterns

### 2. Decision Tree
**Observations**:
- Perfect 100% accuracy achieved
- Successfully learned the hierarchical decision rules
- Excellent interpretability - can visualize decision paths
- No overfitting observed despite tree complexity
- Effectively captures non-linear relationships in the data

### 3. KNN
**Observations**:
- Achieved perfect 100% accuracy with k=5 neighbors
- Distance-based classification works excellently
- Feature normalization was crucial for good performance
- Low computational complexity during training
- Effective for this multi-class problem with well-separated classes

### 4. Naive Bayes
**Observations**:
- Perfect 100% accuracy despite conditional independence assumption
- The dataset features are reasonably separable even with independence assumption
- Gaussian variant works well with continuous features
- Very fast training and prediction
- Robust probabilistic approach

### 5. Random Forest (Ensemble)
**Observations**:
- Perfect 100% accuracy achieved
- Ensemble of 100 trees provides excellent generalization
- No overfitting despite multiple deep trees
- Feature importance readily available
- Combines strengths of multiple decision trees

### 6. XGBoost (Ensemble)
**Observations**:
- Highest accuracy: 99.58% (nearly perfect)
- Gradient boosting approach effective for this problem
- Slight margin of error suggests very challenging classification
- Superior to single decision tree through boosting mechanism
- AUC Score of 1.0 indicates excellent class separation

---

## 💡 Key Insights

### Best Performing Model
- **Model Name**: All models (5 with 100%, 1 with 99.58%)
- **Accuracy**: 99.58% - 100%
- **Reason for Performance**: The synthetic dataset has well-separated classes with distinctive feature patterns:
  - Horses have significantly larger body measurements
  - Cats have sharper claws and more whiskers
  - Dogs have different ear shapes and bite force

### Model Comparison Summary
- **Fastest Models**: Logistic Regression, Naive Bayes
- **Most Accurate (100%)**: Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest
- **Most Balanced (F1 Score)**: All traditional models score 1.0, XGBoost: 0.9958
- **Most Reliable (AUC)**: All models achieve AUC = 1.0 or very close

### Key Findings:
1. **Perfect Separability**: The dataset features are highly discriminative for the three animal classes
2. **Model Consensus**: 5 out of 6 models achieved perfect accuracy, indicating clear decision boundaries
3. **Ensemble Methods**: Both Random Forest and XGBoost show excellent performance
4. **Simplicity Works**: Simple models (Logistic Regression, Naive Bayes) perform as well as complex ones
5. **Feature Engineering**: The 15 carefully chosen features provide excellent discrimination

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip or conda

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository_url>
   cd MLAssignment2_latest
   ```

2. **Create virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train models** (if not already trained)
   ```bash
   python train_models.py
   ```

5. **Run Streamlit app**
   ```bash
   streamlit run app.py
   ```

---

## 📁 Project Structure

```
MLAssignment2_latest/
├── app.py                          # Streamlit web application
├── train_models.py                 # Model training script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── model_results.csv              # Results/metrics from trained models
└── models/                         # Directory containing saved models
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    ├── scaler.pkl
    └── label_encoder.pkl
```

---

## �️ Troubleshooting Guide

### Common Issues & Solutions

#### Issue 1: "ModuleNotFoundError: No module named 'pandas'"
**Solution:**
```bash
# Reinstall all dependencies
pip install -r requirements.txt

# Or install individual package
pip install pandas
```

#### Issue 2: "Port 8501 already in use"
**Solution:**
```bash
# Use a different port
streamlit run app.py --server.port 8502

# Or kill the process using the port
# Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Mac/Linux:
lsof -i :8501
kill -9 <PID>
```

#### Issue 3: "ModuleNotFoundError: No module named 'xgboost'"
**Solution:**
```bash
# XGBoost requires separate installation
pip install xgboost
```

#### Issue 4: Models not found / Streamlit can't load models
**Solution:**
```bash
# Retrain models
python train_models.py

# Verify models folder exists
ls -la models/  # Mac/Linux
dir models/    # Windows
```

#### Issue 5: CSV file upload error
**Solution:**
- Verify CSV has exactly 15 columns
- Check column names match exactly
- Ensure no missing values in data
- Try with the sample data provided

#### Issue 6: "PermissionError: [Errno 13]" when saving models
**Solution:**
```bash
# Run with administrator privileges
# Or change directory permissions
# Make sure models/ directory is writable
```

---

## 📁 Project File Structure

```
MLAssignment2_latest/
│
├── app.py                              # Streamlit web application
│   ├── Home page - project overview
│   ├── Model Evaluation - metrics comparison
│   └── Prediction - CSV upload & predictions
│
├── train_models.py                     # Model training pipeline
│   ├── Data loading & preprocessing
│   ├── Model training (6 models)
│   ├── Evaluation on test set
│   └── Model serialization
│
├── create_dataset.py                   # Dataset generation
│   └── Synthetic data for 3 animal classes
│
├── requirements.txt                    # Package dependencies
│
├── README.md                           # This documentation
│
├── animal_classification_dataset.csv   # Training dataset (1,200 samples)
│
├── model_results.csv                   # Evaluation metrics summary
│
└── models/                             # Saved trained models
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    ├── scaler.pkl                      # Feature scaler
    └── label_encoder.pkl               # Label encoder for predictions
```

---

## 📊 Results & Performance

### Model Performance Summary

All 6 models trained successfully with excellent results:

| Model | Accuracy | AUC | Status |
|---|---|---|---|
| Logistic Regression | 100% | 1.0 | ✅ Perfect |
| Decision Tree | 100% | 1.0 | ✅ Perfect |
| KNN | 100% | 1.0 | ✅ Perfect |
| Naive Bayes | 100% | 1.0 | ✅ Perfect |
| Random Forest | 100% | 1.0 | ✅ Perfect |
| XGBoost | 99.58% | 1.0 | ✅ Excellent |

### Key Metrics Explained

- **Accuracy**: Percentage of correct predictions
- **AUC**: Area under ROC curve (0-1, higher is better)
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of Precision and Recall
- **MCC**: Matthews Correlation Coefficient (-1 to 1)

---

## 🔗 Deployment to Streamlit Cloud

### Prerequisites
- GitHub account
- Git installed
- Code pushed to GitHub

### Deployment Steps

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Go to Streamlit Cloud:**
   - Visit https://streamlit.io/cloud
   - Click "Sign in with GitHub"
   - Authorize Streamlit

3. **Deploy App:**
   - Click "New app"
   - Select your repository
   - Select branch: `main`
   - Set main file path: `app.py`
   - Click "Deploy"

4. **Access Your App:**
   - Streamlit generates a unique URL
   - Share with others
   - App is live in minutes!

**Example URL:** `https://ml-animals-classifier.streamlit.app`

---

## 📚 References

- **Scikit-learn**: https://scikit-learn.org/stable/
- **XGBoost**: https://xgboost.readthedocs.io/
- **Streamlit**: https://docs.streamlit.io/
- **Pandas**: https://pandas.pydata.org/
- **NumPy**: https://numpy.org/

---

## 👨‍💻 Author & Course Info

**Project Title**: ML Classification - Cats, Dogs & Horses  
**Date Created**: January 2026  
**Python Version**: 3.8+  
**License**: MIT

---

## ❓ Quick Reference Commands

```bash
# Setup
python -m venv venv
venv\Scripts\activate              # Windows
source venv/bin/activate           # Mac/Linux
pip install -r requirements.txt

# Generate data
python create_dataset.py

# Train models
python train_models.py

# Run web app
streamlit run app.py

# Access app
# Open: http://localhost:8501
```

---

## ⚖️ Academic Integrity

This project is submitted as original coursework. All code is custom-written following the assignment requirements. The synthetic dataset is generated programmatically, and model training is performed from scratch.

**Anti-Plagiarism Statement:**
- Code: Original implementation
- Dataset: Synthetically generated
- Results: Unique to this training run
- UI: Custom Streamlit app

---

**Last Updated**: January 25, 2026
**Status**: ✅ Complete & Ready for Submission

```
