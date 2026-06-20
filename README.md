# Breast Cancer Classification — SVM vs KNN

A machine learning project that classifies breast tumors as **benign** or **malignant** using the Wisconsin Breast Cancer dataset. The project compares four SVM kernel variants against a K-Nearest Neighbors classifier to find the best-performing model for this binary classification task.

## Overview

Early and accurate diagnosis of breast cancer is critical for treatment outcomes. This project applies classical machine learning algorithms to a well-known diagnostic dataset, evaluating how different models and kernel functions handle the classification of cell nuclei measurements as cancerous or non-cancerous.

## Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic) dataset** (`data.csv`), which contains:

- **569 samples**, each representing a tumor
- **30 numeric features** computed from digitized images of fine needle aspirate (FNA) cell nuclei — including radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension (each measured as mean, standard error, and "worst" value)
- **1 target column** (`diagnosis`): `M` (Malignant) or `B` (Benign)
- **1 identifier column** (`id`), dropped before modeling

Class distribution: 357 benign (62.7%) vs. 212 malignant (37.3%) — a mild class imbalance.

> **Note:** `data.csv` is not included in this repository. Download it from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) or [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) and place it in the working directory (or update the file path in the notebook).

## Project Workflow

1. **Data Loading & Cleaning** — Load `data.csv`, drop the stray unnamed column caused by a trailing comma in the header
2. **Preprocessing** — Check for missing values; encode the `diagnosis` column (`M → 1`, `B → 0`) using `LabelEncoder`
3. **Feature/Target Split** — Drop `id` and `diagnosis` to form the feature matrix `X` (30 features); `diagnosis` becomes target `y`
4. **Train/Test Split** — 80/20 split with `stratify=y` to preserve class proportions (455 train / 114 test samples)
5. **Feature Scaling** — Standardize features with `StandardScaler` (mean = 0, std = 1), since SVM and KNN are distance-based and sensitive to feature scale
6. **Model Training** — Train and evaluate:
   - SVM with 4 kernels: `linear`, `rbf`, `sigmoid`, `poly`
   - KNN with `k=5`
7. **Evaluation** — Accuracy, confusion matrix, and classification report (precision/recall/F1) for each model
8. **Visualization** — Confusion matrix heatmaps for all 5 models side by side

## Results

| Model            | Accuracy   |
|-------------------|-----------:|
| SVM (RBF)         | **97.37%** |
| SVM (Linear)      | 96.49%     |
| KNN (k=5)         | 95.61%     |
| SVM (Sigmoid)     | 94.74%     |
| SVM (Polynomial)  | 88.60%     |

**Best model: SVM with RBF kernel**, achieving 97.4% accuracy with only 3 misclassifications out of 114 test samples (3 false negatives, 0 false positives).

### Why this matters

In a medical diagnostic setting, **false negatives** (a malignant tumor predicted as benign) are far more costly than false positives, since they risk a missed diagnosis. The RBF and linear SVM kernels produced zero false positives and the fewest false negatives, making them the strongest candidates for this task — while the polynomial kernel struggled, missing 13 malignant cases.

## Tech Stack

- Python 3
- [pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) — data handling
- [scikit-learn](https://scikit-learn.org/) — modeling (`SVC`, `KNeighborsClassifier`, `StandardScaler`, `LabelEncoder`, metrics)
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) — visualization

## Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the notebook

1. Clone this repository
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```
2. Download `data.csv` (see [Dataset](#dataset) above) and place it in the project directory
3. Update the file path in the notebook if needed (it currently points to `/content/data.csv`, a Google Colab path)
4. Open and run `Cancer_Classification.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab

## Project Structure

```
.
├── Cancer_Classification.ipynb   # Main notebook: full pipeline
├── data.csv                      # Dataset (not included, see above)
└── README.md
```

## Future Improvements

- Hyperparameter tuning (e.g. `GridSearchCV` for SVM's `C`, `gamma`, and KNN's `k`)
- Cross-validation for more robust performance estimates
- Feature selection/dimensionality reduction (e.g. PCA) given the high feature count
- Try additional models (Random Forest, Logistic Regression, Gradient Boosting) for comparison
- ROC-AUC curves alongside confusion matrices

## License

This project is open source and available under the [MIT License](LICENSE).
