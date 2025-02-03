# 🔷 Machine-Learning: Supervised-SemiSupervised-Unsupervised-ActiveLearning-SVM

## 🔷 Supervised, Semi-Supervised, and Unsupervised Learning | Active Learning Using Support Vector Machines

### 🔶 Overview
This project explores different learning paradigms: **Supervised Learning, Semi-Supervised Learning, and Unsupervised Learning**. It focuses on **Active Learning with Support Vector Machines (SVMs)** to efficiently label data for model training. Various clustering methods and decision boundary analyses are performed to evaluate learning strategies.

---

## 🔷 Datasets Used
- **Breast Cancer Wisconsin (Diagnostic) Data Set**  
  - Binary classification problem (Benign vs. Malignant).
  - Used for supervised, semi-supervised, and unsupervised learning.
  - **[Dataset Link](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data)**

- **Banknote Authentication Data Set**  
  - Binary classification problem for distinguishing genuine vs. forged banknotes.
  - Used for **Active Learning experiments**.
  - **[Dataset Link](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)**

---

## 🔷 Libraries Used
- `numpy` - Numerical computations.
- `pandas` - Data manipulation.
- `matplotlib` & `seaborn` - Data visualization.
- `scikit-learn` - Machine learning models (SVM, k-means, spectral clustering).
- `scipy` - Mathematical functions for clustering.
- `imbalanced-learn` - Handling class imbalance.
- `statsmodels` - Statistical analysis.
- `tqdm` - Progress tracking for Monte Carlo simulations.

---

## 🔷 Steps Taken to Accomplish the Project

### 🔶 1. Supervised Learning with L1-Penalized SVM
- Trained an **L1-penalized SVM** on the **Breast Cancer dataset** for binary classification.
- Used **5-fold cross-validation** to **tune the penalty parameter (C)**.
- **Normalized** the dataset before training.
- **Monte Carlo Simulation:** Repeated the training process **30 times**, averaging performance metrics.
- Reported:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - AUC (Area Under the Curve)
  - Confusion matrices
  - **ROC Curve** for visualization.

---

### 🔶 2. Semi-Supervised Learning (Self-Training)
- Used **50% labeled data** from both classes, treating the rest as **unlabeled**.
- Trained an **L1-penalized SVM** on the labeled subset.
- Selected the **unlabeled point farthest from the decision boundary**, let the model label it, and **added it back** to training data.
- Repeated until all unlabeled points were classified.
- Evaluated final SVM on test data and computed the **same performance metrics as in supervised learning**.

---

### 🔶 3. Unsupervised Learning with K-Means Clustering
- Applied **k-means clustering (k=2)** on the **entire training set** (ignoring labels).
- Ran **multiple iterations** of k-means to avoid local minima.
- Determined **cluster labels** by polling the **30 closest points** to each cluster center.
- Compared k-means predicted labels against true labels.
- Evaluated:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - AUC
  - **ROC Curve**

---

### 🔶 4. Spectral Clustering
- Implemented **Spectral Clustering using RBF kernel**.
- Found an **optimal gamma value** to maintain balance in clusters.
- Assigned labels using **fit-predict method** instead of cluster proximity.
- Compared results with k-means clustering and SVM.

---

### 🔶 5. Active Learning Using SVMs
- **Dataset:** Banknote Authentication Data.
- Compared **Active Learning vs. Passive Learning** using **90 different SVMs**.
- **Passive Learning Approach:**
  - Started with **10 random data points**.
  - Incrementally added **10 more randomly selected points** at each step.
  - Trained SVM iteratively until **all 900 training points** were used.

- **Active Learning Approach:**
  - Started with **10 random data points**.
  - Chose the **10 closest points to the SVM decision boundary** and added them to the training set.
  - Repeated until all training data was used.
- **Final Comparison:**
  - Computed test errors across **50 trials** for both passive and active learning.
  - Plotted **test error vs. number of training samples** for both approaches.

---
## 📌 **Note**
This repository contains **Jupyter Notebooks** detailing each step, along with **results and visualizations**.
