<div align="center">

# 🧠 Learn Machine Learning

### Every Algorithm. From Scratch. With the Math That Actually Explains It.

> _Textbooks give you definitions. This repo gives you intuition — backed by code, visual breakdowns, and hand-written notes._

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![NumPy](https://img.shields.io/badge/NumPy-Arrays-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Pandas](https://img.shields.io/badge/Pandas-DataFrames-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Plots-11557C?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org)
[![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep_Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-Neural_Nets-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Status](https://img.shields.io/badge/Status-Actively_Updated-00C853?style=for-the-badge)](https://github.com/your-username/Learn_Machine_Learning)

</div>

---

## 📌 What It Does

**Learn_Machine_Learning** is a structured, notebook-first deep dive into machine learning — built for people who want to actually understand what's happening under the hood, not just call `.fit()` and move on.

Every algorithm in this repo comes with:

- 📖 **Plain-English intuition** — what the algorithm is *actually* doing
- 📐 **The math** — core equations explained step by step, not hidden
- ✍️ **Handwritten notes** — personal notes scanned for visual learners who think better on paper
- 💻 **From-scratch implementation** — coded in pure Python/NumPy before using sklearn
- 🔬 **sklearn comparison** — validate the scratch version against the library
- 📊 **Visual output** — decision boundaries, loss curves, confusion matrices, cluster plots

This is not a tutorial copy-paste repo. This is a **reference-grade ML knowledge base** built while learning deeply — and it shows.

---

## 🗺️ Learning Path & Curriculum

```
Learn_Machine_Learning/
│
├── 00_Foundations/
│   ├── numpy_essentials.ipynb
│   ├── pandas_for_ml.ipynb
│   ├── matplotlib_seaborn.ipynb
│   └── math_for_ml/
│       ├── linear_algebra.ipynb
│       ├── calculus_gradients.ipynb
│       └── probability_statistics.ipynb
│
├── 01_Supervised_Learning/
│   │
│   ├── Regression/
│   │   ├── 01_linear_regression/
│   │   │   ├── theory.ipynb          ← intuition + math
│   │   │   ├── scratch.ipynb         ← numpy from scratch
│   │   │   ├── sklearn_impl.ipynb    ← library version
│   │   │   └── notes/               ← 📝 handwritten notes
│   │   │
│   │   ├── 02_polynomial_regression/
│   │   ├── 03_ridge_lasso/           ← L1, L2 regularization
│   │   └── 04_elastic_net/
│   │
│   ├── Classification/
│   │   ├── 05_logistic_regression/
│   │   ├── 06_knn/
│   │   ├── 07_naive_bayes/
│   │   ├── 08_svm/                   ← kernel trick explained
│   │   ├── 09_decision_tree/         ← Gini, entropy, info gain
│   │   └── 10_random_forest/
│   │
│   └── Ensemble/
│       ├── 11_bagging/
│       ├── 12_boosting/              ← AdaBoost + Gradient Boosting
│       └── 13_xgboost/
│
├── 02_Unsupervised_Learning/
│   ├── 14_kmeans/                    ← elbow method, inertia
│   ├── 15_hierarchical_clustering/   ← dendrograms
│   ├── 16_dbscan/
│   ├── 17_pca/                       ← eigenvalues, explained variance
│   └── 18_tsne/
│
├── 03_Model_Evaluation/
│   ├── bias_variance_tradeoff.ipynb
│   ├── cross_validation.ipynb
│   ├── metrics_classification.ipynb  ← precision, recall, F1, ROC-AUC
│   ├── metrics_regression.ipynb      ← MAE, MSE, RMSE, R²
│   └── hyperparameter_tuning.ipynb   ← GridSearch, RandomSearch
│
├── 04_Feature_Engineering/
│   ├── encoding_categorical.ipynb
│   ├── scaling_normalization.ipynb
│   ├── handling_missing_data.ipynb
│   └── feature_selection.ipynb
│
├── 05_Deep_Learning/
│   ├── 19_neural_network_scratch/    ← backprop from scratch
│   ├── 20_ann_keras/
│   ├── 21_cnn_basics/
│   └── 22_rnn_lstm/
│
└── handwritten_notes/               ✍️ Scanned personal notes
    ├── linear_regression_notes.pdf
    ├── logistic_regression_notes.pdf
    ├── svm_notes.pdf
    ├── decision_tree_notes.pdf
    ├── neural_network_notes.pdf
    └── ...
```

---

## 🏛️ How Each Algorithm Is Taught

```
For every algorithm, the structure is identical ↓

  ┌─────────────────────────────────────────────────────────┐
  │                   ALGORITHM BREAKDOWN                   │
  ├─────────────────────────────────────────────────────────┤
  │                                                         │
  │  STEP 1 → INTUITION                                     │
  │  "What problem does this solve? What's the core idea?"  │
  │                                                         │
  │  STEP 2 → MATH                                          │
  │  Core equation → gradient → update rule                 │
  │  Written out with explanation, not just LaTeX drops     │
  │                                                         │
  │  STEP 3 → FROM SCRATCH (Pure NumPy)                     │
  │  class LinearRegression:                                │
  │      def fit(self, X, y): ...                           │
  │      def predict(self, X): ...                          │
  │                                                         │
  │  STEP 4 → sklearn VALIDATION                            │
  │  Verify scratch output matches sklearn — if it does,    │
  │  the math is right.                                     │
  │                                                         │
  │  STEP 5 → VISUALIZATION                                 │
  │  Decision boundaries, loss curves, residual plots       │
  │                                                         │
  │  STEP 6 → ✍️ HANDWRITTEN NOTES                         │
  │  Personal scanned notes — for review without a screen   │
  │                                                         │
  └─────────────────────────────────────────────────────────┘
```

---

## ✍️ Handwritten Notes Preview

> **Replace this section** with actual scanned note photos/screenshots.

```
📸  Suggested setup:
    1. Scan or photograph your clearest 2–3 note pages per topic
    2. Crop, adjust contrast (free on phone: Adobe Scan / CamScanner)
    3. Drop into handwritten_notes/ folder in the repo
    4. Add a 2×2 photo grid here using Markdown image syntax

Example layout (replace with real images):

|  Linear Regression Notes  |  Neural Network Notes  |
|:-------------------------:|:----------------------:|
|  ![](handwritten_notes/lr_notes.jpg)  |  ![](handwritten_notes/nn_notes.jpg)  |

|  SVM Kernel Notes  |  Decision Tree Notes  |
|:-----------------:|:--------------------:|
|  ![](handwritten_notes/svm_notes.jpg)  |  ![](handwritten_notes/dt_notes.jpg)  |
```

The handwritten notes are what make this repo different from every other "ML from scratch" repo on GitHub. **Show them prominently.**

---

## 📊 Algorithms Covered

### Supervised Learning — Regression

| Algorithm | From Scratch | sklearn | Notes | Visualizations |
|---|---|---|---|---|
| Linear Regression | ✅ | ✅ | ✍️ | Loss curve, regression line |
| Polynomial Regression | ✅ | ✅ | ✍️ | Overfitting demo |
| Ridge Regression (L2) | ✅ | ✅ | ✍️ | Regularization path |
| Lasso Regression (L1) | ✅ | ✅ | ✍️ | Feature sparsity plot |
| Elastic Net | ✅ | ✅ | ✍️ | — |

### Supervised Learning — Classification

| Algorithm | From Scratch | sklearn | Notes | Visualizations |
|---|---|---|---|---|
| Logistic Regression | ✅ | ✅ | ✍️ | Decision boundary, sigmoid |
| K-Nearest Neighbors | ✅ | ✅ | ✍️ | K vs accuracy plot |
| Naive Bayes | ✅ | ✅ | ✍️ | Probability tables |
| Support Vector Machine | ✅ | ✅ | ✍️ | Kernel trick, margin |
| Decision Tree | ✅ | ✅ | ✍️ | Tree visualization, Gini |
| Random Forest | ✅ | ✅ | ✍️ | Feature importance |

### Ensemble Methods

| Algorithm | From Scratch | sklearn | Notes | Key Concept |
|---|---|---|---|---|
| Bagging | ✅ | ✅ | ✍️ | Bootstrap aggregation |
| AdaBoost | ✅ | ✅ | ✍️ | Weak learner weighting |
| Gradient Boosting | ✅ | ✅ | ✍️ | Residual fitting |
| XGBoost | — | ✅ | ✍️ | Regularized boosting |

### Unsupervised Learning

| Algorithm | From Scratch | sklearn | Notes | Visualizations |
|---|---|---|---|---|
| K-Means | ✅ | ✅ | ✍️ | Elbow curve, cluster plot |
| Hierarchical Clustering | — | ✅ | ✍️ | Dendrogram |
| DBSCAN | — | ✅ | ✍️ | Noise point detection |
| PCA | ✅ | ✅ | ✍️ | Explained variance plot |
| t-SNE | — | ✅ | ✍️ | High-dim visualization |

### Model Evaluation & Feature Engineering

| Topic | Covered | Key Content |
|---|---|---|
| Bias-Variance Tradeoff | ✅ | Underfitting vs overfitting plots |
| Cross Validation | ✅ | K-Fold, Stratified K-Fold |
| Classification Metrics | ✅ | Precision, Recall, F1, ROC-AUC, Confusion Matrix |
| Regression Metrics | ✅ | MAE, MSE, RMSE, R² |
| Hyperparameter Tuning | ✅ | GridSearchCV, RandomizedSearchCV |
| Feature Encoding | ✅ | One-Hot, Label, Ordinal |
| Scaling | ✅ | StandardScaler, MinMaxScaler, RobustScaler |
| Missing Data | ✅ | Imputation strategies |

---

## ⚡ How to Run

### Prerequisites

- Python 3.9+
- pip

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/Learn_Machine_Learning.git
cd Learn_Machine_Learning
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter

```bash
jupyter notebook
```

Navigate to any algorithm folder and open the notebooks in order:
`theory.ipynb` → `scratch.ipynb` → `sklearn_impl.ipynb`

### 4. Start Here (Recommended Path)

```
If you're new to ML:
  00_Foundations → 01_Supervised/Regression/01_linear_regression

If you know the basics:
  Jump directly to any algorithm folder

If you're interview prepping:
  03_Model_Evaluation → read all handwritten_notes/
```

### Packages Installed

```
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
tensorflow
keras
xgboost
scipy
```

---

## 💡 Why This Repo Is Different

```
Most ML repos:                       This repo:

✗ Copy-paste from tutorials    vs    ✅ Written while actually learning
✗ Just sklearn .fit() calls    vs    ✅ From-scratch before sklearn
✗ No math explanation          vs    ✅ Math → code → visualization
✗ No personal touch            vs    ✅ Handwritten notes included
✗ Bloated, disorganized        vs    ✅ Identical structure per algorithm
✗ Static, abandoned            vs    ✅ Actively updated
```

---

## 🗺️ Roadmap

- [x] Supervised Learning (Regression + Classification)
- [x] Ensemble Methods
- [x] Unsupervised Learning
- [x] Model Evaluation & Feature Engineering
- [x] Handwritten notes for all major algorithms
- [ ] Natural Language Processing section
- [ ] Time Series Forecasting section
- [ ] End-to-end ML project walkthroughs
- [ ] Cheatsheet PDFs per algorithm

---

## 👤 Author

**Rohan Sharma** — AI/ML Engineer · LangGraph Developer  
MCA @ GL Bajaj Institute of Technology & Management | BCA @ GGSIPU — 9.5 CGPA

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/rohan-sharma-048266246)
[![Email](https://img.shields.io/badge/Email-sharma1718rohan@gmail.com-D14836?style=flat-square&logo=gmail)](mailto:sharma1718rohan@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/your-username)

---

<div align="center">

**⭐ Star this repo if it helped you actually understand ML — not just use it**

_If you find a bug in a from-scratch implementation, open an issue. That's how we both learn._

</div>
