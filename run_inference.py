import os
import sys
import numpy as np
import joblib
import zipfile

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

def load_processed_data():
    print("Loading processed data...")
    X_test_1 = np.load("data/processed/X_test_1_selected.npy")
    X_test_2 = np.load("data/processed/X_test_2_unlabeled.npy")
    print(f"Test set 1 shape: {X_test_1.shape}")
    print(f"Test set 2 shape: {X_test_2.shape}")
    return X_test_1, X_test_2

def generate_predictions(model, X_test_1, X_test_2):
    print("\nGenerating test set predictions...")
    test1_proba = model.predict_proba(X_test_1)
    if uses_cuml and isinstance(test1_proba, cupy.ndarray):
        test1_proba = test1_proba.get()
    print(f"Test set 1 prediction probability shape: {test1_proba.shape}")
    test2_proba = model.predict_proba(X_test_2)
    if uses_cuml and isinstance(test2_proba, cupy.ndarray):
        test2_proba = test2_proba.get()
    print(f"Test set 2 prediction probability shape: {test2_proba.shape}")
    os.makedirs("preds", exist_ok=True)
    np.save("preds/preds_1.npy", test1_proba)
    np.save("preds/preds_2.npy", test2_proba)
    print("Prediction results saved to models/preds_1.npy and models/preds_2.npy")
    return test1_proba, test2_proba

def create_zipfile():
    print("\nCreating Zipfile...")
    if not os.path.exists("preds/preds_1.npy") or not os.path.exists("preds/preds_2.npy"):
        print("Prediction files do not exist, cannot create Zipfile")
        return
    with zipfile.ZipFile("pred_results.zip", "w") as zipf:
        zipf.write("preds/preds_1.npy", "preds_1.npy")
        zipf.write("preds/preds_2.npy", "preds_2.npy")
    print("Zipfile created: pred_results.zip")

def main():
    if not os.path.exists('data/processed/X_test_1_selected.npy') or not os.path.exists('data/processed/X_test_2_unlabeled.npy'):
        print("Processed data does not exist, please run process_test.py first")
        return

    default_model_name = "XGBoost"
    model_name = ''
    model = None
    if len(sys.argv) == 2 and sys.argv[1] in model_file[True].keys():
        model_name = sys.argv[1]
    elif len(sys.argv) == 1:
        model_name = default_model_name

    if not uses_cuml and model_name == "SVM":
        print(f'Selected unsupport model, using default {default_model_name} instead')
        print('Loading model...')
        model = joblib.load(os.path.join("models", model_file[True][default_model_name]))
    elif model_name != '' and os.path.exists(os.path.join("models", model_file[True][model_name])):
        fl = model_name == default_model_name
        print(f"{'Selected' if not fl else 'Default'} model: {model_name}{' (best model)' if fl else ''}")
        print('Loading model...')
        model = joblib.load(os.path.join("models", model_file[True][model_name]))
    elif os.path.exists(os.path.join("models", model_file[True][default_model_name])):
        if model_name == '':
            print(f'Selected unknown model, using default {default_model_name} instead')
            print('Loading model...')
        elif model_name != '':
            print(f'Selected model {model_name} does not exist, using default {default_model_name} instead')
            print('Loading model...')
        model = joblib.load(os.path.join("models", model_file[True][default_model_name]))
    else:
        print('Neither the selected model nor the default model exists')
        print('Please run resample.py and train_models.py first')
        return
    
    X_test_1, X_test_2 = load_processed_data()
    test1_proba, test2_proba = generate_predictions(model, X_test_1, X_test_2)
    create_zipfile()

if __name__ == "__main__":
    main()