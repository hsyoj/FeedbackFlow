import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from collections import Counter
from sklearn.decomposition import PCA

# Set random seed for reproducibility
np.random.seed(42)

def load_data():
    """Load datasets"""
    print("Loading train datasets...")
    
    # Load training data
    X_train = pd.read_csv("data/raw/X_train.csv")
    y_train = pd.read_csv("data/raw/y_train.csv")
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    
    return X_train, y_train

def explore_data(X_train, y_train):
    """Perform exploratory data analysis"""
    print("\nStarting exploratory data analysis...")
    
    # Display basic feature statistics
    print("\nFeature summary statistics:")
    print(X_train.describe().T.head())
    
    # Check for missing values
    print("\nChecking for missing values:")
    missing_values = X_train.isnull().sum()
    print(f"Total missing values in features: {missing_values.sum()}")
    
    if missing_values.sum() > 0:
        print("Features with missing values:")
        print(missing_values[missing_values > 0])
    
    # Analyze label distribution
    print("\nAnalyzing label distribution:")
    label_counts = y_train.iloc[:, 0].value_counts().sort_index()
    print(label_counts)
    
    # Compute class imbalance ratio
    max_class_count = label_counts.max()
    min_class_count = label_counts.min()
    imbalance_ratio = max_class_count / min_class_count
    print(f"\nClass imbalance ratio (max/min): {imbalance_ratio:.2f}")
    
    # Plot class distribution
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=label_counts.index, y=label_counts.values)
    plt.title("Training Set Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    for i, count in enumerate(label_counts.values):
        ax.text(i, count + 5, str(count), ha="center")
    plt.tight_layout()
    os.makedirs("reports/figures", exist_ok=True)
    plt.savefig("reports/figures/class_distribution.png")
    plt.close()
    
    # Compute feature correlations for the first 20 features
    print("\nComputing feature correlations...")
    corr_matrix = X_train.iloc[:, :20].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap (First 20 Features)")
    plt.tight_layout()
    plt.savefig("reports/figures/feature_correlation.png")
    plt.close()
    
    return label_counts

def preprocess_data(X_train, y_train):
    """Preprocess data: scaling and feature selection"""
    print("\nStarting train data preprocessing...")
    
    # Flatten labels into a 1D array
    y_train = y_train.iloc[:, 0].values
    
    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Feature selection
    print("Selecting top features...")
    selector = SelectKBest(f_classif, k=250)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    
    print(f"Number of selected features: {X_train_selected.shape[1]}")
    
    # Gather feature scores
    selected_indices = selector.get_support(indices=True)
    feature_scores = selector.scores_
    
    feature_importance = pd.DataFrame({
        'Feature': [f'Feature_{i}' for i in range(len(feature_scores))],
        'Score': feature_scores
    })
    
    top_features = feature_importance.sort_values('Score', ascending=False).head(20)
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Score', y='Feature', data=top_features)
    plt.title('Top 20 Features by F-score')
    plt.tight_layout()
    plt.savefig("reports/figures/feature_importance.png")
    plt.close()
    
    # Save preprocessing objects
    os.makedirs("models", exist_ok=True)
    import joblib
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(selector, "models/selector.pkl")
    print("Preprocessing objects saved to models/scaler.pkl and models/selector.pkl")
    
    return X_train_selected, y_train, scaler, selector

def handle_imbalance_improved(X_train_selected, y_train):
    """Handle class imbalance using SMOTE and Tomek Links"""
    print("\nHandling class imbalance with improved method...")
    
    # Display original class distribution
    print("Original class distribution:")
    original_counter = Counter(y_train)
    print(original_counter)
    
    # Define total target samples
    total_samples = 10000
    target_samples_per_class = total_samples // len(original_counter)
    print(f"Target average samples per class: {target_samples_per_class}")
    
    # Build sampling strategy: only upsample minority classes
    sampling_strategy = {}
    for label, count in original_counter.items():
        sampling_strategy[label] = max(count, target_samples_per_class)
    
    print("Sampling strategy:")
    for label, count in sorted(sampling_strategy.items()):
        print(f"Class {label}: {count} samples")
    
    # Upsample minority classes
    print("Applying SMOTE...")
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled_smote, y_resampled_smote = smote.fit_resample(X_train_selected, y_train)
    
    # Clean boundary samples using Tomek Links
    print("Applying Tomek Links cleaning...")
    tomek = TomekLinks(sampling_strategy='all')
    X_resampled, y_resampled = tomek.fit_resample(X_resampled_smote, y_resampled_smote)
    
    # Show new class distribution
    print("Resampled class distribution:")
    resampled_counter = Counter(y_resampled)
    print(resampled_counter)
    print(f"Total samples after resampling: {sum(resampled_counter.values())}")
    
    # Plot before vs after class distribution
    plt.figure(figsize=(12, 6))
    df = pd.DataFrame({
        'Original': pd.Series(original_counter),
        'SMOTE+Tomek': pd.Series(resampled_counter)
    })
    df.plot(kind='bar', figsize=(12, 6))
    plt.title('Class Distribution Comparison: Original vs SMOTE+Tomek')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("reports/figures/class_distribution_comparison_improved.png")
    plt.close()
    
    evaluate_resampling_quality(X_train_selected, y_train, X_resampled, y_resampled)
    
    return X_resampled, y_resampled

def evaluate_resampling_quality(X_original, y_original, X_resampled, y_resampled):
    """Evaluate quality of resampling via PCA visualization and class change percentages"""
    print("\nEvaluating resampling quality...")
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    X_original_pca = pca.fit_transform(X_original)
    X_resampled_pca = pca.transform(X_resampled)
    
    # Plot PCA scatter for original and resampled data
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    for label in np.unique(y_original):
        idx = y_original == label
        plt.scatter(X_original_pca[idx, 0], X_original_pca[idx, 1], alpha=0.5,
                    label=f'Class {label}' if label < 5 else None)
    plt.title('Original Data (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    
    plt.subplot(1, 2, 2)
    for label in np.unique(y_resampled):
        idx = y_resampled == label
        plt.scatter(X_resampled_pca[idx, 0], X_resampled_pca[idx, 1], alpha=0.5,
                    label=f'Class {label}' if label < 5 else None)
    plt.title('Resampled Data (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    
    plt.tight_layout()
    plt.savefig("reports/figures/resampling_quality_pca.png")
    plt.close()
    
    # Calculate class count changes
    original_counts = Counter(y_original)
    resampled_counts = Counter(y_resampled)
    change_percentage = {label: ((resampled_counts[label] - original_counts[label]) / original_counts[label]) * 100
                         if original_counts[label] > 0 else float('inf')
                         for label in original_counts}
    change_df = pd.DataFrame({'Class': list(change_percentage.keys()),
                               'Change (%)': list(change_percentage.values())})
    change_df = change_df.sort_values('Change (%)', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Class', y='Change (%)', data=change_df)
    plt.title('Percentage Change in Class Counts After Resampling')
    plt.xlabel('Class')
    plt.ylabel('Change (%)')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("reports/figures/class_change_percentage.png")
    plt.close()
    
    print("Resampling quality evaluation complete")

def save_processed_data(X_train_selected, y_train, X_resampled, y_resampled):
    """Save processed data to disk"""
    print("\nSaving processed data...")
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/X_train_selected.npy", X_train_selected)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/X_resampled.npy", X_resampled)
    np.save("data/processed/y_resampled.npy", y_resampled)
    print("Train data processing and saving complete!")

def main():
    """Main execution function"""
    X_train, y_train = load_data()
    explore_data(X_train, y_train)
    X_train_selected, y_train_vals, _, _ = preprocess_data(X_train, y_train)
    X_resampled, y_resampled = handle_imbalance_improved(X_train_selected, y_train_vals)
    save_processed_data(X_train_selected, y_train_vals, X_resampled, y_resampled)
    print("\nResampling and test set 2 processing complete!")

if __name__ == "__main__":
    main()