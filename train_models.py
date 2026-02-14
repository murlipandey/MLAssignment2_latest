"""
ML Classification Models Training Script
Trains 6 different classification models and evaluates them using multiple metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, auc, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report, roc_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

class MLClassificationPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        return self.df
    
    def preprocess_data(self, target_column):
        """Preprocess the data - handle missing values, encode categorical variables"""
        print("\nPreprocessing data...")
        
        # Drop missing values
        self.df = self.df.dropna()
        
        # Separate features and target
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        
        # Encode categorical features if any
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        
        # Encode target variable
        self.le_target = LabelEncoder()
        y_encoded = self.le_target.fit_transform(y)
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Feature scaling
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        
    def train_models(self):
        """Train all 6 classification models"""
        print("\n" + "="*60)
        print("TRAINING CLASSIFICATION MODELS")
        print("="*60)
        
        # 1. Logistic Regression
        print("\n1. Training Logistic Regression...")
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = lr
        
        # 2. Decision Tree Classifier
        print("2. Training Decision Tree Classifier...")
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(self.X_train, self.y_train)
        self.models['Decision Tree'] = dt
        
        # 3. K-Nearest Neighbor Classifier
        print("3. Training K-Nearest Neighbor...")
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(self.X_train, self.y_train)
        self.models['KNN'] = knn
        
        # 4. Naive Bayes Classifier
        print("4. Training Naive Bayes Classifier...")
        nb = GaussianNB()
        nb.fit(self.X_train, self.y_train)
        self.models['Naive Bayes'] = nb
        
        # 5. Random Forest Classifier
        print("5. Training Random Forest (Ensemble)...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        
        # 6. XGBoost Classifier
        print("6. Training XGBoost (Ensemble)...")
        xgb = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
        xgb.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = xgb
        
        print("\n✓ All models trained successfully!")
        
    def evaluate_models(self):
        """Evaluate all models using multiple metrics"""
        print("\n" + "="*60)
        print("EVALUATING MODELS")
        print("="*60)
        
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            
            # Predictions
            y_pred = model.predict(self.X_test)
            
            # For probability-based metrics
            try:
                y_pred_proba = model.predict_proba(self.X_test)
                # Handle binary and multi-class
                if y_pred_proba.shape[1] == 2:
                    y_pred_proba = y_pred_proba[:, 1]
            except:
                y_pred_proba = None
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            mcc = matthews_corrcoef(self.y_test, y_pred)
            
            # AUC Score
            try:
                if len(np.unique(self.y_test)) == 2:  # Binary classification
                    auc_score = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else accuracy
                else:  # Multi-class
                    auc_score = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr') if y_pred_proba is not None else accuracy
            except:
                auc_score = accuracy
            
            self.results[model_name] = {
                'Accuracy': accuracy,
                'AUC': auc_score,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'MCC': mcc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  AUC: {auc_score:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  MCC: {mcc:.4f}")
    
    def save_models(self, models_dir='models'):
        """Save trained models to disk"""
        print(f"\n\nSaving models to {models_dir}/...")
        for model_name, model in self.models.items():
            model_path = f"{models_dir}/{model_name.replace(' ', '_').lower()}.pkl"
            joblib.dump(model, model_path)
            print(f"  ✓ Saved {model_name} to {model_path}")
        
        # Save scaler
        joblib.dump(self.scaler, f"{models_dir}/scaler.pkl")
        joblib.dump(self.le_target, f"{models_dir}/label_encoder.pkl")
        print(f"  ✓ Saved scaler and label encoder")
    
    def generate_results_summary(self):
        """Generate a summary of results"""
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        
        # Create results dataframe
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.drop(columns=['y_pred', 'y_pred_proba'])
        
        print("\nMetrics Comparison Table:")
        print(results_df.round(4))
        
        # Save results
        results_df.to_csv('model_results.csv')
        print("\n✓ Results saved to model_results.csv")
        
        return results_df
    
    def run_pipeline(self, target_column, csv_path=None):
        """Run the entire pipeline"""
        if csv_path:
            self.data_path = csv_path
        
        self.load_data()
        self.preprocess_data(target_column)
        self.train_models()
        self.evaluate_models()
        self.save_models()
        results_df = self.generate_results_summary()
        
        return results_df


# Example usage:
if __name__ == "__main__":
    # Replace with your dataset path
    csv_path = "animal_classification_dataset.csv"
    target_column = "Animal_Type"  # Replace with your target column name
    
    pipeline = MLClassificationPipeline(csv_path)
    results = pipeline.run_pipeline(target_column)
