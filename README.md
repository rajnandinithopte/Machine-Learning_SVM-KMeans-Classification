# 🔷 Machine-Learning: SVM-KMeans-Classification

## 🔷 Multi-Class and Multi-Label Classification with Support Vector Machines and K-Means Clustering

### 🔶 Overview
This project focuses on **Support Vector Machines (SVM)** and **K-Means Clustering** for both **multi-class** and **multi-label classification**. It applies **kernel methods, feature scaling, and clustering techniques** to datasets from the **UCI Machine Learning Repository**. The primary objective is to explore **hyperparameter tuning, data preprocessing, and clustering evaluation metrics** to achieve optimal classification performance.

## 🔷 Datasets Used

### 📌 Breast Cancer Wisconsin (Diagnostic) Dataset  
- **Source:** [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))  
- Contains **features extracted from cell nuclei in breast cancer biopsies**.
- The classification goal is to distinguish between **benign and malignant tumors**.

### 📌 Banknote Authentication Dataset  
- **Source:** [UCI Repository](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)  
- Contains **image-based statistical features of banknotes**.
- Used to classify banknotes as **genuine or forged** based on **Wavelet Transform features**.

## 🔷 Libraries Used
- `pandas` - Data manipulation and preprocessing.
- `numpy` - Numerical computations.
- `matplotlib` & `seaborn` - Data visualization.
- `sklearn.svm` - Support Vector Machine (SVM) modeling.
- `sklearn.cluster` - K-Means clustering implementation.
- `sklearn.preprocessing` - Feature scaling and transformation.
- `sklearn.metrics` - Model evaluation metrics.
- `scipy` - Statistical analysis.

## 🔷 Steps Taken to Accomplish the Project

### 🔶 1. Data Preprocessing and Feature Engineering
- Handled **missing values** and performed **feature scaling** using **Standardization (Z-score normalization)**.
- Encoded categorical labels for **multi-class classification**.
- Split the datasets into **training and testing sets**.

### 🔶 2. Exploratory Data Analysis (EDA)
- Created **pairplots and histograms** to analyze feature distributions.
- Used **correlation heatmaps** to identify relationships between features.
- Applied **Principal Component Analysis (PCA)** to visualize data separability.

### 🔶 3. Support Vector Machine (SVM) Classification
- Implemented **linear and non-linear SVM classifiers** using **RBF and polynomial kernels**.
- Used **grid search and cross-validation** to tune **C (regularization parameter)** and **γ (kernel coefficient)**.
- Compared **test accuracy and confusion matrices** across different kernel types.

### 🔶 4. K-Means Clustering
- Applied **K-Means clustering** for unsupervised classification.
- Used the **Elbow Method** and **Silhouette Score** to determine the optimal number of clusters.
- Visualized clustering results using **2D and 3D plots**.

### 🔶 5. Evaluation Metrics
- Computed **Precision, Recall, and F1-score** for SVM classification.
- Evaluated clustering quality using **Silhouette Score and Davies-Bouldin Index**.
- Compared clustering performance against actual class labels.

### 🔶 6. Hyperparameter Tuning
- Performed **grid search** to optimize hyperparameters for SVM.
- Experimented with **different kernel functions** and **regularization strengths**.
- Fine-tuned **K-Means cluster initialization strategies**.

### 🔶 7. Final Model Selection and Analysis
- Selected the best-performing **SVM model** based on test accuracy.
- Identified **feature importance** and **misclassification patterns**.
- Analyzed **clustering performance against supervised learning models**.

  ---
## 📌 **Note**
This repository contains **Jupyter Notebooks** detailing each step, along with **results and visualizations**.
