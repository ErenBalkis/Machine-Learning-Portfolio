# Machine Learning Portfolio 🚀

This repository documents my learning journey in machine learning, containing projects, data analysis, and code implementations. It is designed to bridge the gap between theoretical knowledge and practical "hands-on" experience.

## 📂 Repository Structure

Projects are categorized by machine learning subfields:

* **01_Supervised_Learning:** Supervised learning algorithms (Classification, Regression, etc.)
* *(Upcoming)* **02_Unsupervised_Learning:** Unsupervised learning (Clustering, etc.)

---

## 🔬 Featured Project: Breast Cancer Classification with k-NN

This is the first project in this portfolio, addressing a medical diagnostic problem: **Breast Cancer Classification**.

🔗 **View Project:** [01_KNN_Breast_Cancer_Classification.ipynb](./01_Supervised_Learning/01_KNN_Breast_Cancer_Classification.ipynb)

### 📌 Project Summary
The goal of this project is to develop a machine learning model to predict whether a tumor is **Malignant** or **Benign** using the **Wisconsin Breast Cancer Dataset**.

### 🛠️ Technologies & Libraries Used
* **Language:** Python 3.x
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn (sklearn)

### 📊 Project Steps
The following data science steps were implemented in this notebook:

1.  **Exploratory Data Analysis (EDA):** Analyzed class balance, missing values, and feature correlations (Heatmap).
2.  **Data Preprocessing:**
    * Split the dataset into Training (80%) and Test (20%) sets (using `stratify`).
    * Applied **StandardScaler** for feature scaling, as k-NN is a distance-based algorithm.
3.  **Model Training & Tuning:**
    * Implemented the **k-Nearest Neighbors (k-NN)** algorithm.
    * Performed hyperparameter optimization by testing values from 1 to 20 to find the optimal 'k' neighbors.
4.  **Evaluation:**
    * Evaluated model performance using **Confusion Matrix** and **Classification Report**.
    * Focused on **Recall** score (alongside Accuracy) to minimize false negatives (missed cancer cases), which is critical in medical diagnostics.

---
### 2. Handwriting Recognition with SVM ✍️
**File:** [`SVM_Classification.ipynb`](SVM_Classification.ipynb)

In this project, I built a Support Vector Machine (SVM) model to classify handwritten digits using the **Scikit-learn Digits Dataset**. The project focuses on image classification fundamentals and model evaluation.

* **Objective:** To correctly identify digits (0-9) from 8x8 pixel grayscale images.
* **Methodology:**
    * Loaded and visualized the dataset using `matplotlib`.
    * Preprocessed data by flattening 8x8 image matrices into 1D vectors.
    * Split the data into training and testing sets (80/20 split).
    * Trained a **Support Vector Classifier (SVC)** with a linear kernel.
* **Results:**
    * Evaluated performance using a **Confusion Matrix** (visualized with `seaborn`) and a **Classification Report**.
    * Achieved high accuracy across all digit classes.
* **Tech Stack:** Python, Scikit-learn, Matplotlib, Seaborn.

---

## 🛠️ Tools & Libraries
The projects in this portfolio primarily use the following tools:
* **Python** (NumPy, Pandas)
* **Machine Learning:** Scikit-learn
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebook / Anaconda

## 📬 Contact

Feel free to reach out for questions or feedback.

* **GitHub:** [ErenBalkis](https://github.com/ErenBalkis)
* **LinkedIn:** [ErenBalkis](https://linkedin.com/in/eren-balkis)
