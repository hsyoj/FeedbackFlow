import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

# Set random seed for reproducibility
np.random.seed(42)

def load_data():
    """Load datasets"""
    print("Loading test datasets...")
    
    # Load test set 1
    X_test_1 = pd.read_csv("data/raw/X_test_1.csv")
    
    # Load test set 2 and its partial labels
    X_test_2 = pd.read_csv("data/raw/X_test_2.csv")
    y_test_2_reduced = pd.read_csv("data/raw/y_test_2_reduced.csv")
    
    print(f"Test set 1 shape: {X_test_1.shape}")
    print(f"Test set 2 shape: {X_test_2.shape}")
    print(f"Test set 2 partial labels shape: {y_test_2_reduced.shape}")
    
    return X_test_1, X_test_2, y_test_2_reduced

def preprocess_data(X_test_1, X_test_2):
    """Preprocess data: scaling and feature selection"""
    print("\nStarting test data preprocessing...")
    
    # Standardize features
    print("Standardizing features...")
    scaler = joblib.load("models/scaler.pkl")
    X_test_1_scaled = scaler.transform(X_test_1)
    X_test_2_scaled = scaler.transform(X_test_2)

    # Feature selection
    print("Performing feature selection...")
    selector = joblib.load("models/selector.pkl")
    X_test_1_selected = selector.transform(X_test_1_scaled)
    X_test_2_selected = selector.transform(X_test_2_scaled)

    return X_test_1_selected, X_test_2_selected

def process_test2(X_test_2_selected, y_test_2_reduced):
    """Process test set 2 by excluding the first labeled samples"""
    print("\nProcessing test set 2: excluding initial labeled samples...")
    total_samples = X_test_2_selected.shape[0]
    labeled_samples = y_test_2_reduced.shape[0]
    unlabeled_samples = total_samples - labeled_samples
    print(f"Total samples in test set 2: {total_samples}")
    print(f"Labeled samples: {labeled_samples}")
    print(f"Unlabeled samples: {unlabeled_samples}")
    X_test_2_unlabeled = X_test_2_selected[labeled_samples:]
    print(f"Unlabeled data shape: {X_test_2_unlabeled.shape}")
    return X_test_2_unlabeled

def save_processed_data(X_test_1_selected, X_test_2_selected, X_test_2_unlabeled):
    """Save processed data to disk"""
    print("\nSaving processed data...")
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/X_test_1_selected.npy", X_test_1_selected)
    np.save("data/processed/X_test_2_selected.npy", X_test_2_selected)
    np.save("data/processed/X_test_2_unlabeled.npy", X_test_2_unlabeled)
    print("Test data processing and saving complete!")

def main():
    """Main execution function"""
    if not os.path.exists('models/scaler.pkl') or not os.path.exists('models/selector.pkl'):
        print("Preprocessing models do not exist, please run resample.py first")
        return
    
    X_test_1, X_test_2, y_test_2_reduced = load_data()
    X_test_1_selected, X_test_2_selected = preprocess_data(X_test_1, X_test_2)
    X_test_2_unlabeled = process_test2(X_test_2_selected, y_test_2_reduced)
    save_processed_data(X_test_1_selected, X_test_2_selected, X_test_2_unlabeled)
    print("\nResampling and test set 2 processing complete!")

if __name__ == "__main__":
    main()