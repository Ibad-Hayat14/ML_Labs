#  Machine Learning Labs (F23 AI - Green)

> **Student**: Ibadullah Hayat (Reg No: B23F0001AI010)  
> **Instructor**: Mr Usama Ahmed Khan
> **University**: Pak-Austria Fachhochschule Institute of Applied Sciences and Technology 
> **Course**: Machine Learning (Fall 2025)

This repository contains the complete set of **12 hands-on Machine Learning labs**, covering foundational concepts from data preprocessing to advanced modeling techniques. Each lab is implemented using **Python, NumPy, Pandas, Scikit-learn, and Matplotlib**, with clear explanations and visualizations.

All notebooks are self-contained, well-documented, and ready to run perfect for learning, review, or portfolio.

---

##  Lab Overview

| Lab | Title | Key Concepts |
|-----|-------|--------------|
| **Lab 01** | Introduction to NumPy & Pandas | Array operations, Series, DataFrame, filtering, aggregation |
| **Lab 02** | Data Wrangling & Visualization | Missing value handling, EDA, scatter/bar/histogram plots |
| **Lab 03** | Linear Regression on Insurance Data | Feature scaling, train-test split, gradient descent, evaluation |
| **Lab 04** | Gradient Descent Variants | Batch, Stochastic (SGD), Mini-batch GD comparison |
| **Lab 05** | Logistic Regression (Breast Cancer) | Binary classification, confusion matrix, precision/recall, ROC |
| **Lab 06** | Support Vector Machine (SVM) | Linear/RBF kernels, hyperparameter tuning, SMOTE not used |
| **Lab 07** | K-Nearest Neighbors (KNN) | Optimal `k` selection, distance metrics, manual KNN example |
| **Lab 08** | Decision Trees | Tree visualization, Gini vs. Entropy, feature importance |
| **Lab 09** | Random Forest | Ensemble learning, out-of-bag (OOB) score, hyperparameter tuning |
| **Lab 10** | Multi-Layer Perceptron (MLP) | Neural networks, activation functions (ReLU, tanh, sigmoid), regularization |
| **Lab 11** | Unsupervised Learning (K-Means + PCA) | Clustering, Elbow method, dimensionality reduction, cluster interpretation |
| **Lab 12** | End-to-End ML Pipeline (Titanic) | Full pipeline: imputation, encoding, SMOTE, PCA, ElasticNet, hyperparameter tuning |

---
```
## Repository Structure
ml-labs/
├── Lab01_NumPy_Pandas_Fundamentals.ipynb
├── Lab02_Data_Wrangling_Visualization.ipynb
├── Lab03_Linear_Regression_Insurance.ipynb
├── Lab04_Gradient_Descent_Variants.ipynb
├── Lab05_Logistic_Regression_Breast_Cancer.ipynb
├── Lab06_SVM_Breast_Cancer.ipynb
├── Lab07_KNN_Breast_Cancer.ipynb
├── Lab08_Decision_Trees_Wine.ipynb
├── Lab09_Random_Forest_Wine.ipynb
├── Lab10_MLP_Wine.ipynb
├── Lab11_Unsupervised_Learning_Wine.ipynb
├── Lab12_End_to_End_Pipeline_Titanic.ipynb
└── README.md
```
 Note: Datasets are either loaded via sklearn.datasets or included in standard libraries (e.g., seaborn.load_dataset). No external data files are required.
---

# How to run
### Clone the repo
git clone https://github.com/your-username/ml-labs.git
cd ml-labs

### Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn scipy imbalanced-learn

### Launch Jupyter
jupyter notebook
---

 ### Key Skills Demonstrated
 Data Preprocessing: Cleaning, encoding, scaling, handling missing values
 Exploratory Data Analysis (EDA): Visualizations, correlation, distributions
 Model Implementation: From scratch (GD) and using scikit-learn
 Evaluation Metrics: Accuracy, precision, recall, F1, ROC-AUC, confusion matrix
 Hyperparameter Tuning: GridSearchCV, RandomizedSearchCV
 Advanced Techniques: SMOTE, PCA, ElasticNet, ensemble methods
 Pipeline Building: Robust, reusable ML workflows

 ### License
This work is for educational purposes. Feel free to use it as a reference or learning resource.

Portfolio Tip: This repo demonstrates a progressive learning journey - from basic data manipulation to full ML pipelines—ideal for internships or graduate applications.
