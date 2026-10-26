import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import time

uses_cuml = True
try:
    from cuml.svm import SVC
    import cupy
    import cudf
except ImportError:
    uses_cuml = False

if uses_cuml:
    cupy.random.seed(42)

np.random.seed(42)

def load_processed_data():
    print("Loading processed data...")
    X_train = np.load("data/processed/X_train_selected.npy")
    y_train = np.load("data/processed/y_train.npy")
    X_resampled = np.load("data/processed/X_resampled.npy")
    y_resampled = np.load("data/processed/y_resampled.npy")
    print(f"Original training set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Resampled training set shape: X={X_resampled.shape}, y={y_resampled.shape}")
    return X_train, y_train, X_resampled, y_resampled

def create_validation_set(X, y, test_size=0.2, random_state=42):
    print("\nCreating validation set...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation set shape: X={X_val.shape}, y={y_val.shape}")
    return X_train, X_val, y_train, y_val

def define_models():
    print("\nDefining multiple machine learning models...")
    models = {
        "LogisticRegression": LogisticRegression(
            solver="lbfgs", 
            max_iter=1000, 
            C=1.0,
            random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            random_state=42,
            n_jobs=-1
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multiclass",
            random_state=42,
            verbose=-1,
            n_jobs=-1
        ),
        "CatBoost": cb.CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            loss_function="MultiClass",
            random_seed=42,
            verbose=0
        ),
        "NeuralNetwork": MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size=256,
            learning_rate="adaptive",
            max_iter=200,
            random_state=42
        ),
        "SVM": SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            random_state=42
        ) if uses_cuml else None,
    }
    return models

model_file = {
    True:{
    "LogisticRegression": "LogisticRegression_resampled.pkl",
    "RandomForest": "RandomForest_resampled.pkl",
    "SVM": "SVM_resampled.pkl",
    "XGBoost": "XGBoost_resampled.pkl",
    "LightGBM": "LightGBM_resampled.pkl",
    "CatBoost": "CatBoost_resampled.pkl",
    "NeuralNetwork": "NeuralNetwork_resampled.pkl"
    },
    False:{
    "LogisticRegression": "LogisticRegression_original.pkl",
    "RandomForest": "RandomForest_original.pkl",
    "SVM": "SVM_original.pkl",
    "XGBoost": "XGBoost_original.pkl",
    "LightGBM": "LightGBM_original.pkl",
    "CatBoost": "CatBoost_original.pkl",
    "NeuralNetwork": "NeuralNetwork_original.pkl"
    }
}

def train_and_evaluate_models(models, X_train, y_train, X_val, y_val, use_resampled=True):
    data_type = "resampled" if use_resampled else "original"
    print(f"\nTraining and evaluating models (using {data_type} data)...")
    os.makedirs("models", exist_ok=True)
    results = {}
    for name in models.keys():
        if not uses_cuml and name == "SVM":
            continue
        model_exists = os.path.exists(os.path.join("models", model_file[use_resampled][name]))
        model = None
        start_time = time.time()
        if model_exists:
            model = joblib.load(os.path.join("models", model_file[use_resampled][name]))
        else:
            model = models[name]
            print(f"\nTraining model: {name}")
            start_time = time.time()
            if name == 'SVM':
                X_train_cp = cupy.asarray(X_train)
                y_train_cp = cupy.asarray(y_train)
                model.fit(X_train_cp, y_train_cp)
            else:
                model.fit(X_train, y_train)
        if name == 'SVM':
            y_pred_cp = model.predict(X_val)
            y_pred_proba_cp = model.predict_proba(X_val)
            y_pred = cupy.asnumpy(y_pred_cp)
            y_pred_proba = cupy.asnumpy(y_pred_proba_cp)
        else:
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        wce_loss = weighted_cross_entropy_loss(y_val, y_pred_proba)
        precision_macro = precision_score(y_val, y_pred, average="macro", zero_division=0)
        recall_macro = recall_score(y_val, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)
        precision_weighted = precision_score(y_val, y_pred, average="weighted", zero_division=0)
        recall_weighted = recall_score(y_val, y_pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_val, y_pred, average="weighted", zero_division=0)
        conf_matrix = confusion_matrix(y_val, y_pred)
        fig, ax = plt.subplots(figsize=(15, 12))
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=(model.classes_ if hasattr(model, "classes_") else None))
        disp.plot(cmap=plt.cm.PuBuGn, ax=ax, values_format='d')
        ax.set_xlabel('Predicted Label', fontsize=20, labelpad=20)
        ax.set_ylabel('True Label', fontsize=20, labelpad=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        for txt in disp.text_.ravel():
            txt.set_fontsize(15)
        plt.title(f"Confusion Matrix - {name}", fontsize=20, pad=20)
        plt.tight_layout()
        plt.savefig(f"reports/figures/{name}_confusion_matrix.png")
        plt.close()
        try:
            roc_auc = roc_auc_score(y_val, y_pred_proba, multi_class="ovr", average="macro")
        except:
            roc_auc = None
        report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
        training_time = time.time() - start_time
        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "weighted_cross_entropy_loss": wce_loss,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "roc_auc": roc_auc,
            "report": report,
            "conf_matrix": conf_matrix,
            "training_time": training_time
        }
        if not model_exists:
            print(f"{name} training completed, time taken: {training_time:.2f} seconds")
        else:
            print(f"{name} loaded from existing model, time taken: {training_time:.2f} seconds")
        print(f"Validation set weighted cross entropy loss: {wce_loss:.6f}")
        print(f"Validation set accuracy: {accuracy:.4f}")
        print(f"Validation set macro F1 score: {f1_macro:.4f}")
        print(f"Validation set weighted F1 score: {f1_weighted:.4f}")
        if not model_exists:
            model_filename = os.path.join("models", model_file[use_resampled][name])
            joblib.dump(model, model_filename)
            print(f"Model saved to: {model_filename}")
        print()
    return results

def weighted_cross_entropy_loss(y_true, y_pred_proba):
    n_samples = len(y_true)
    n_classes = y_pred_proba.shape[1]
    y_true_one_hot = np.zeros((n_samples, n_classes))
    for i, label in enumerate(y_true):
        y_true_one_hot[i, label] = 1
    class_counts = np.sum(y_true_one_hot, axis=0)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / np.sum(class_weights)
    sample_weights = np.sum(y_true_one_hot * class_weights, axis=1)
    epsilon = 1e-15
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    log_probs = np.log(y_pred_proba)
    weighted_loss = -np.mean(sample_weights * np.sum(y_true_one_hot * log_probs, axis=1))
    return weighted_loss

def compare_models(results, metric="weighted_cross_entropy_loss", ascending=True):
    print(f"\nComparing model performance (using {metric} metric)...")
    model_names = list(results.keys())
    metrics = [results[name][metric] for name in model_names]
    comparison_df = pd.DataFrame({
        "Model": model_names,
        metric: metrics
    }).sort_values(by=metric, ascending=ascending)
    print(comparison_df)
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="Model", y=metric, data=comparison_df)
    plt.title(f"Model Performance Comparison ({metric})")
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    for i, m in enumerate(metrics):
        ax.text(i, m + 0.01, f"{m:.6f}", ha="center")
    plt.tight_layout()
    os.makedirs("reports/figures", exist_ok=True)
    plt.savefig(f"reports/figures/model_comparison_{metric}.png")
    plt.close()
    return comparison_df

def main():
    if not os.path.exists('data/processed/X_train_selected.npy') or not os.path.exists('data/processed/y_train.npy') or \
       not os.path.exists('data/processed/X_resampled.npy') or not os.path.exists('data/processed/y_resampled.npy'):
        print("Processed data not found, please run resample.py first.")
        return

    X_train, y_train, X_resampled, y_resampled = load_processed_data()
    X_train_resampled, X_val_resampled, y_train_resampled, y_val_resampled = create_validation_set(X_resampled, y_resampled)
    #X_train_orignal, X_val_orignal, y_train_orignal, y_val_orignal = create_validation_set(X_train, y_train)
    models = define_models()
    results = train_and_evaluate_models(models, X_train_resampled, y_train_resampled, X_val_resampled, y_val_resampled, use_resampled=True)
    #results = train_and_evaluate_models(models, X_train_orignal, y_train_orignal, X_val_orignal, y_val_orignal, use_resampled=False)
    comparison_df = compare_models(results, metric="weighted_cross_entropy_loss", ascending=True)
    best_model_name = comparison_df.iloc[0]["Model"]
    print(f"\nBest model: {best_model_name}")
    print("\nModel implementation and evaluation completed!")

if __name__ == "__main__":
    main()