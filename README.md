# FeedbackFlow
## About The Project

FeedbackFlow is a machine learning project designed to streamline customer feedback management in manufacturing. By leveraging advanced multiclass classification techniques, it accurately categorizes feedback across 28 product-specific departments, optimizing workflow and enhancing response efficiency. The project tackles challenges like class imbalance and distribution shift, delivering robust models for real-world deployment.

### Built With

These libraries and tools enable the execution of most models in the project.

| **Category**         | **Technologies**                             |
| -------------------- | -------------------------------------------- |
| Programming Language | Python (>=3.10)                              |
| Core ML Libraries    | scikit-learn, xgboost, catboost, lightgbm    |
| Data Preprocessing   | numpy, pandas, imblearn (SMOTE, Tomek Links) |
| Visualization        | matplotlib, seaborn                          |
| Optimization Tools   | joblib                                       |

To run all models, including those with advanced features or GPU acceleration, install these additional libraries. 

| **Category**              | **Technologies**                                           |
| ------------------------- | ---------------------------------------------------------- |
| CUDA Toolkits             | CUDA (>=11.8), cuDNN (>=8.0)                               |
| GPU Acceleration          | RAPIDS (cuml, cupy, cudf)                                  |
| **Hardware Requirements** | **Nvidia RTX/Quadro GPU with >= 8GB VRAM**                 |
| **OS Requirements**       | **Linux, WSL2 (Windows Subsystem for Linux) on Win 11/10** |



## Content

* **[Getting Start](#Getting-Start)**
  * [Environment Preparation](#Environment-Preparation)
    * [Windows](#Windows)
    * [macOS](#macOS)
    * [Linux](#Linux)
  * [Additional Preparation on Windows](#Additional-Preparation-on-Windows)
    * [Install WSL2](#Install-WSL2)
    * [Install Nvidia Driver](#Install-Nvidia-Driver)
    * [Install CUDA Toolkit](#Install-CUDA-Toolkit)
    * [Prepare Environment](#Prepare-Environment)
  * [Requirements Installation](#Requirements-Installation)
    * [Using Pip](#Using-Pip)
    * [Using Conda](#Using-Conda)
  * [Datasets Preparation](#Datasets-Preparation)
    * [Set Folder Permissions](#Set-Folder-Permissions)
    * [Organize Dataset](#Organize-Dataset)
  * [Running the Project](#Running-the-Project)
    * [Prerequisites](#Prerequisites)
    * [Running Scripts](#Running-Scripts)
* **[Evaluation on Models](#Evaluation-on-Models)**
  * [Model Overview](#Model-Overview)
  * [Model Performance Metrics](#Model-Performance-Metrics)



## Getting Start

### Environment Preparation

Before getting started, make sure your system has Python installed along with a package manager (either pip or conda). Follow the instructions below according to your operating system.

#### Windows

- **Python**: Download and install Python from the [official website](https://www.python.org/downloads/windows/), or install it via the [Microsoft Store](https://www.microsoft.com/store/productId/9NRWMJP3717K).

- **Pip**: Python installed via python.org comes with pip. To install or upgrade pip separately, refer to the [official guide](https://pip.pypa.io/en/stable/installation/).

- **Conda**: Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) for a more comprehensive package manager.

#### MacOS

- **Python**: Install Python via [Homebrew](https://brew.sh/) using the command `brew install python`, or download it from the [official website](https://www.python.org/downloads/macos/).

- **Pip**: Python from python.org includes pip. For separate installation or upgrade, see the [official guide](https://pip.pypa.io/en/stable/installation/).

- **Conda**: Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

#### Linux

- **Python**: Python is typically pre-installed. To install or upgrade, use your distribution's package manager (e.g., on Debian/Ubuntu, run `sudo apt install python3`).
- **Pip**: Install pip via your package manager (e.g., on Debian/Ubuntu, run `sudo apt install python3-pip`). For more options, see the [official guide](https://pip.pypa.io/en/stable/installation/).
- **Conda**: Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

### Additional Preparation on Windows

To run all models on Windows **10** and **11**, the environment should be configured as follows:

#### Install WSL2

Open **PowerShell** as Administrator, and run the following command to install WSL2 along with Ubuntu 24.04:

```bash
wsl --install --distribution Ubuntu-24.04
```

If you encounter an error indicating that Ubuntu-24.04 is not found, you may need to update WSL first:

```bash
wsl --update
```

After updating, run the installation command again.

If WSL is not yet enabled on your system, you can manually enable it with the following commands:

```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

Then restart your computer and proceed with the installation.

To check the installed distributions and the current WSL version, use:

```bash
wsl --list --verbose
```

Ensure that your distribution is set to version 2:

```bash
wsl --set-version Ubuntu-24.04 2
```

#### Install Nvidia Driver

Ensure your Windows system has the latest NVIDIA GPU driver that supports WSL2.

1. Download the latest driver from the [NVIDIA official website](https://www.nvidia.com/Download/index.aspx).
2. Install the driver normally by following the on-screen instructions.

Make sure the driver version is **465** or later for CUDA support inside WSL2.

#### Install CUDA Toolkit

Once inside the Ubuntu-24.04 WSL environment, follow these steps:

1. Add the CUDA repository:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /"
```

2. Install the CUDA Toolkit:

```bash
sudo apt update
sudo apt install -y cuda-toolkit-12-4
```

3. (Optional) Add CUDA to PATH:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

You can verify the installation by running:

```bash
nvcc --version
```

#### Prepare Environment

Following the steps described in [Environment Preparation/Linux](#Linux) to set up the environment.

### Requirements Installation

Once Python and a package manager are ready, install the required libraries for the project. You may choose to use either pip or conda.

#### Using Pip

Install basic dependencies with pip:

```bash
pip install numpy pandas imbalanced-learn
pip install scikit-learn xgboost catboost lightgbm
pip install matplotlib seaborn
pip install joblib
```

To install additional dependencies, make sure the CUDA is properly installed.

For CUDA 12:

```bash
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==25.4.*" "dask-cudf-cu12==25.4.*" "cuml-cu12==25.4.*" \
    "cugraph-cu12==25.4.*" "nx-cugraph-cu12==25.4.*" "cuspatial-cu12==25.4.*" \
    "cuproj-cu12==25.4.*" "cuxfilter-cu12==25.4.*" "cucim-cu12==25.4.*" \
    "pylibraft-cu12==25.4.*" "raft-dask-cu12==25.4.*" "cuvs-cu12==25.4.*" \
    "nx-cugraph-cu12==25.4.*"
```

For CUDA 11.8:

```bash
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu11==25.4.*" "dask-cudf-cu11==25.4.*" "cuml-cu11==25.4.*" \
    "cugraph-cu11==25.4.*" "nx-cugraph-cu11==25.4.*" "cuspatial-cu11==25.4.*" \
    "cuproj-cu11==25.4.*" "cuxfilter-cu11==25.4.*" "cucim-cu11==25.4.*" \
    "pylibraft-cu11==25.4.*" "raft-dask-cu11==25.4.*" "cuvs-cu11==25.4.*" \
    "nx-cugraph-cu11==25.4.*"
```

Or following the [official guideline](https://docs.rapids.ai/install/).

#### Using Conda

Install basic with conda, specifying channels where needed:

```bash
conda install numpy pandas scikit-learn matplotlib seaborn joblib
conda install -c conda-forge imbalanced-learn
conda install -c conda-forge xgboost catboost lightgbm
```

To install additional dependencies, make sure the CUDA is properly installed.

Following is an example for `CUDA 12.0~12.8` and `python 3.12`:

```bash
conda -c rapidsai -c conda-forge -c nvidia  \
    rapids=25.04 python=3.12 'cuda-version>=12.0,<=12.8'
```

CUDA version options are: `cuda-version>=12.0,<=12.8` and `'cuda-version>=11.4,<=11.8'`; Python version options are: `python 3.10`, `python 3.11`, and `python 3.12`. Select the appropriate versions for installation based on your specific requirements or refer to the [official guideline](https://docs.rapids.ai/install/).

**Note**: Some libraries may require additional channels or specific installation steps. If you encounter any issues, please refer to the official documentation of the respective libraries. 

**Note:** Although RAPIDS officially provides installation packages for CUDA versions below 11.8, to ensure **optimal performance, compatibility, and stability**, avoid using versions lower than 11.8 whenever possible.

After completing the steps above, your environment should be successfully configured to run the this project.

### Datasets Preparation

To get started with this project, you need to prepare the dataset. The following steps guide you to put the dataset correctly.

#### Set Folder Permissions

To ensure your project files and datasets are accessible without permission issues, set the proper permissions on **macOS, WSL, or Linux**.

Open your terminal, navigate to your project root folder, and run:

```bash
chmod -R 755 your_project_directory
```

Replace `your_project_directory` with the actual path to your project folder.

This command recursively (`-R`) sets read and execute permissions for everyone and write permissions for the owner.

#### Organize Dataset

For **macOS, WSL, or Linux**, create the following directory structure inside your project folder:

```bash
mkdir -p data/raw
```

For **Windows**, use File Explorer to create two folders manually:

1. `data`
2. `data/raw` inside `data`

Then, move your datasets into the `data/raw` folder.

Your `data/raw` folder must contain the following files:

- `X_train.csv` (**Required**)
- `y_train.csv` (**Required**)
- `X_test_1.csv` (Required for testing)
- `X_test_2.csv` (Required for testing)
- `y_test_2_reduced.csv` (Required for testing)

The structure should look like this:

```plain
ProjectFolder/
├── data/
│   └── raw/
│       ├── X_train.csv
│       ├── y_train.csv
│       └── ...
├── process_test.py
├── README.md
├── resample.py
├── run_inference.py
└── train_models.py
```

**Note**: Make sure the filenames are **exactly** as listed above, since the training and testing scripts expect them. 

### Running the Project

This section provides detailed instructions for running this project, which involves executing multiple scripts to implement and compare machine learning methods for customer feedback classification. The guide covers running each script, understanding the underlying methods, and saving models, tailored to the project's specific needs, such as multiclass classification and handling class imbalance.

#### Prerequisites

* Environment Setup: Ensure the environment is configured as outlined in the [Environment Preparation](#Environment-Preparation) and [Requirements Installation](#Requirements-Installation) sections of the documentation.
* Dataset Preparation: Prepare the dataset as described in the [Datasets Preparation](#Datasets-Preparation) section,

#### Running Scripts

Run the following commands individually in the terminal to execute the Python scripts for training and testing:

* **Training Scripts**:
  
  ```bash
  python resample.py
  python train_models.py
  ```
  * `resample.py`: Applies resampling techniques (SMOTE+Tomek) to address class imbalance in the training set (X_train).
  * `train_models.py`: Trains the classification model on the resampled data using models listed [here](#Model-Overview).
  
* **Testing Scripts (Optional)**:
  
  ```bash
  python process_test.py
  python run_inference.py [model_name]
  ```
  * `process_test.py`: Preprocesses the test sets (X_test1 and X_test2) as needed, potentially adapting to distribution shifts.
  * `run_inference.py`: Evaluates the trained model on the test sets and outputs predictions. It requires one argument: the name of model. Model names are available [here](#Model-Overview). The `model_name` parameter defaults to the tested best model (`XGBoost`). A zip file named `pred_results.zip` will be generated in the project’s root directory after successful execution.

**Note:** Please run the data-processing scripts (`resample.py` & `process_test.py`) ***before*** training the models (`train_models.py`) and generating predictions(`run_inference.py`). In particular, ensure that models training is completed prior to running the prediction step.



## **Evaluation on Models** 
### Model Overview

The models and its hyperparameters used in the `train_models.py` are as follows:

| Model Name         | Hyperparameters                                              |
| ------------------ | ------------------------------------------------------------ |
| LogisticRegression | `solver="lbfgs"`<br />`max_iter=1000`<br />`C=1.0`<br />`random_state=42` |
| RandomForest       | `n_estimators=100`<br />`max_depth=20`<br />`min_samples_split=5`<br />`min_samples_leaf=2`<br />`random_state=42`<br />`n_jobs=-1` |
| XGBoost            | `n_estimators=100`<br />`learning_rate=0.1`<br />`max_depth=6`<br />`subsample=0.8`<br />`colsample_bytree=0.8`<br />`objective="multi:softprob"`<br />`random_state=42`<br />`n_jobs=-1` |
| LightGBM           | `n_estimators=100`<br />`learning_rate=0.1`<br />`max_depth=6`<br />`num_leaves=31`<br />`subsample=0.8`<br />`colsample_bytree=0.8`<br />`objective="multiclass"`<br />`random_state=42`<br />`verbose=-1`<br />`n_jobs=-1` |
| NeuralNetwork      | `hidden_layer_sizes=(100, 50)`<br />`activation="relu"`<br />`solver="adam"`<br />`alpha=0.0001`<br />`batch_size=256`<br />`learning_rate="adaptive"`<br />`max_iter=200`<br />`random_state=42` |
| *SVM*              | `kernel="rbf"`<br />`C=10.0`<br />`gamma="scale"`<br />`probability=True`<br />`random_state=42` |

**Note:** `SVM` only applicable to WSL2 and Linux systems with CUDA environment installed.

### Model Performance Metrics

The following evaluation metrics are derived from the results of model evaluating. These metrics provide a comprehensive assessment of the model's predictive capabilities, including its **weighted cross-entropy loss** (WCE-loss), accuracy, and overall generalization to unseen data.

| Model Name         | WCE-loss     | Accuracy   | F1-score   |
| ------------------ | ------------ | ---------- | ---------- |
| **XGBoost**        | **0.015327** | 0.8724     | 0.8464     |
| SVM                | 0.018065     | **0.8947** | **0.8612** |
| LightGBM           | 0.018331     | 0.8754     | 0.8534     |
| LogisticRegression | 0.019595     | 0.8541     | 0.8226     |
| CatBoost           | 0.023627     | 0.8302     | 0.7855     |
| RandomForest       | 0.026514     | 0.8270     | 0.7804     |
| NeuralNetwork      | 0.030805     | 0.8799     | 0.8603     |

