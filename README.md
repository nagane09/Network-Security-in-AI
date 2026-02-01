# Network Security in AI  
**A Research-Oriented Machine Learning Pipeline for Network Risk Prediction**

🔗 **Live Deployment**  
👉 https://nagane09-network-security-in-ai-app-jbhxfu.streamlit.app/

---

# NetworkSecurity: An End-to-End Machine Learning Pipeline for Network Threat Detection

## Abstract

Network security is a critical domain that aims to protect digital assets from unauthorized access, attacks, and data breaches. Modern network security increasingly relies on data-driven methods to detect anomalies and malicious behaviors in large-scale network traffic. The NetworkSecurity project presents a comprehensive, modular machine learning pipeline that integrates data ingestion, validation, transformation, and model training to predict network security risk scores. This pipeline emphasizes reproducibility, robustness, and interpretability, making it suitable for research, academic, and industrial applications.

## 1. Introduction

The rise of cyber threats such as phishing, malware, and DDoS attacks necessitates intelligent systems capable of analyzing network logs and predicting potential threats. Traditional rule-based methods often fail to capture complex patterns in dynamic network environments. To overcome these limitations, this project implements a machine learning-based approach to analyze network traffic data and produce risk predictions.

This pipeline is designed to:

- Automate data ingestion from MongoDB databases.
- Perform data validation and drift detection.
- Transform raw data into ML-ready formats using imputation and preprocessing.
- Train multiple machine learning models and select the best performing model.
- Enable streamlined deployment via a local Streamlit interface for predictions.

## 2. Project Architecture

The pipeline is composed of four major stages:

### 2.1 Data Ingestion

**Objective:** Collect network traffic data from MongoDB and prepare it for downstream processing.

**Key Steps:**

- **MongoDB Connection:** Using the `pymongo` library, the system connects to the database via a secure URI with TLS verification.
- **Data Extraction:** Collections are exported to a Pandas DataFrame.
- **Cleaning:** Columns like `_id` are removed and placeholder missing values (`"na"`) are converted to `NaN`.
- **Feature Store:** Raw data is saved in a feature store directory to ensure reproducibility.
- **Train-Test Split:** Data is split into training and testing datasets according to a configurable ratio.

**Artifact Generated:** `DataIngestionArtifact` containing paths to training and testing CSV files.

### 2.2 Data Validation

**Objective:** Ensure dataset integrity and detect distribution shifts over time.

**Key Steps:**

- **Schema Validation:** Checks if the number of columns in the dataset matches the expected schema.
- **Data Drift Detection:** Uses the Kolmogorov-Smirnov test to detect statistical changes between training and test datasets.
- **Directory Management:** Organizes validated and invalid datasets into separate directories.
- **Reporting:** Generates a YAML-based drift report for downstream analysis.

**Artifact Generated:** `DataValidationArtifact` with validation status, valid/invalid file paths, and drift report.

### 2.3 Data Transformation

**Objective:** Transform raw data into machine learning-ready formats.

**Key Steps:**

- **Missing Value Imputation:** Utilizes `KNNImputer` to fill missing values based on nearest neighbors.
- **Pipeline Construction:** Wraps preprocessing steps in a Scikit-learn Pipeline to maintain consistency between train and test data.
- **Feature-Target Separation:** The target column is separated and transformed into a binary format.
- **Data Export:** Stores transformed arrays as `.npy` files and saves the fitted preprocessor for inference.

**Artifact Generated:** `DataTransformationArtifact` containing transformed train/test arrays and the preprocessor object.

### 2.4 Model Training

**Objective:** Train multiple machine learning models and select the best performing one.

**Key Steps:**

- **Model Selection:** Evaluates classifiers including:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - AdaBoost
- **Hyperparameter Tuning:** Uses predefined search grids for each model to optimize performance.
- **Metrics Evaluation:** Computes F1 score, precision, and recall on train and test sets.
- **MLflow Tracking (Optional):** Logs metrics and models locally using MLflow for reproducibility.
- **Model Serialization:** Saves the best model and the preprocessor for deployment.

**Artifact Generated:** `ModelTrainerArtifact` including trained model path and classification metrics.

# 3.Model Evaluation Report

## Model: Random Forest Classifier

The Random Forest model was evaluated on the test dataset using the saved model from the `final_model` folder. Feature transformations applied during training were applied consistently to the test data using the saved preprocessor.

### Evaluation Metric

| Metric   | Value |
|---------|-------|
| Accuracy | 0.99  |

> The model achieved **99% accuracy** on the test dataset, indicating that it correctly classified 99% of the network events.



##4. Random Forest: Overview, Working, and Advantages

###  What is Random Forest?

Random Forest is an ensemble machine learning algorithm used for both classification and regression tasks. It combines multiple decision trees to make more accurate and stable predictions. Instead of relying on a single tree (which can overfit), Random Forest builds a “forest” of trees and aggregates their outputs.

###  How Random Forest Works

1. **Bootstrap Sampling:** Multiple subsets of the training data are created by random sampling with replacement.
2. **Decision Tree Construction:** A decision tree is trained on each subset.
3. **Random Feature Selection:** At each split in a tree, a random subset of features is considered, which increases model diversity.
4. **Aggregation:**
   - **Classification:** Each tree votes for a class; the majority vote becomes the final prediction.
   - **Regression:** The average of all tree predictions is taken.



###  Advantages of Random Forest

- **Robustness to Overfitting:** Aggregating multiple trees reduces variance and improves generalization.
- **Handles Missing Data:** Can handle missing values better than many other algorithms.
- **Feature Importance:** Provides insights into which features are most influential.
- **Scalability:** Works well on large datasets and high-dimensional spaces.
- **Non-linear Relationships:** Can capture complex, non-linear interactions in data.

## 4. Conclusion

The NetworkSecurity pipeline provides a robust, end-to-end framework for network threat detection using machine learning. It emphasizes:

- Modularity
- Reproducibility
- Data integrity
- Model performance tracking

Random Forest is one of the key models used in the pipeline due to its accuracy, interpretability, and robustness. This system can be extended to incorporate real-time streaming data, advanced feature engineering, and ensemble learning for higher accuracy in detecting sophisticated cyber threats.


