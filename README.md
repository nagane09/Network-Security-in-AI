# Network Security in AI  
**A Research-Oriented Machine Learning Pipeline for Network Risk Prediction**

🔗 **Live Deployment**  
👉 https://nagane09-network-security-in-ai-app-jbhxfu.streamlit.app/

---

## Abstract

Network security systems increasingly rely on data-driven methods to detect anomalous or malicious behavior in large-scale network traffic. This project presents an **end-to-end, research-oriented machine learning pipeline** for predicting network security risk scores from historical network logs.  

The proposed system integrates **data ingestion from MongoDB**, **schema validation**, **robust preprocessing with missing-value imputation**, and **comparative model evaluation**, culminating in a deployed **Streamlit-based inference application**. Emphasis is placed on **reproducibility, modularity, and quantitative evaluation**, aligning with best practices in applied data science research.

---

## 1. Introduction & Motivation

Modern networks generate massive volumes of heterogeneous logs containing missing values, noise, and schema inconsistencies. Traditional rule-based security systems struggle to scale and adapt to such complexity.  

This project investigates whether **supervised machine learning models**, trained on historical network telemetry, can reliably predict a continuous **network security risk score**, enabling early detection and prioritization of potential threats.

Key research questions addressed:

- Can structured ML pipelines handle real-world, imperfect network data?
- How do ensemble-based models (e.g., XGBoost) compare to classical approaches?
- Can the pipeline be made reproducible and deployable for real-time inference?

---

## 2. System Architecture Overview

The system follows a **modular ML pipeline architecture**, inspired by production-grade and research workflows:

1. Data Ingestion (MongoDB)
2. Data Validation (Schema & consistency checks)
3. Data Transformation (Imputation & preprocessing)
4. Model Training & Evaluation
5. Deployment for Real-Time Inference (Streamlit)

Each stage produces explicit **artifacts**, enabling traceability and reproducibility.

---

## 3. Tech Stack

- **Programming Language:** Python  
- **Core Libraries:** NumPy, Pandas, Scikit-learn  
- **Modeling:** XGBoost  
- **Data Storage:** MongoDB  
- **Experiment Tracking:** MLflow  
- **Deployment:** Streamlit  
- **Version Control:** Git / GitHub  

---

## 4. Evaluation Metrics

Model performance is evaluated using standard regression metrics:

- **Mean Squared Error (MSE)**  
  \[
  \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  \]

- **Mean Absolute Error (MAE)**  
  \[
  \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
  \]

- **Coefficient of Determination (R²)**  
  \[
  R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
  \]

These metrics provide complementary perspectives on prediction accuracy and variance explanation.

---

## 5. Data Ingestion

**File:** `components/data_ingestion.py`

### Objective
Ingest raw network security data from MongoDB and prepare train/test splits.

### Methodology
- Secure MongoDB connection via `pymongo` with TLS certification (`certifi`)
- Conversion of MongoDB collections to Pandas DataFrames
- Removal of database-specific identifiers (`_id`)
- Normalization of missing values (`"na"` → `np.nan`)
- Storage in a local feature store
- Stratified train–test split using Scikit-learn

### Output Artifact
- `DataIngestionArtifact`
  - Train dataset path
  - Test dataset path

---

## 6. Data Validation

**File:** `components/data_validation.py`

### Objective
Ensure structural and semantic integrity of the dataset prior to modeling.

### Validation Checks
- Presence of required input and target columns
- Schema consistency between train and test splits
- Detection of invalid or missing target labels
- Logging of anomalies and inconsistencies

### Output Artifact
- `DataValidationArtifact`
  - Validated train dataset
  - Validated test dataset

---

## 7. Data Transformation

**File:** `components/data_transformation.py`

### Objective
Transform raw tabular data into a model-ready numerical representation.

### Methodology
- Separation of features and target variable
- Target correction (`-1 → 0`) to ensure binary consistency
- Missing-value imputation using **KNNImputer (k = 3)**
- Construction of a Scikit-learn preprocessing pipeline
- Transformation of both training and testing datasets
- Persistence of preprocessing objects for reproducibility

### Output Artifacts
- `transformed_train.npy`
- `transformed_test.npy`
- `preprocessor.pkl`

---

## 8. Model Training & Selection

**File:** `components/model_trainer.py`

### Candidate Models
- Random Forest Regressor
- Gradient Boosting Regressor
- **XGBoost Regressor (Selected)**

### Final Model Configuration

```python
XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    random_state=42
)
````

### Model Selection Rationale

The **XGBoost Regressor** was selected as the final model due to the following properties:

- **Ability to model non-linear feature interactions**, which are common in network traffic data.
- **Robustness to noisy and partially missing data**, especially when combined with prior imputation.
- **Strong empirical performance on structured tabular datasets**, consistently outperforming baseline ensemble models in preliminary experiments.

**Output Artifact**
- `model.pkl` → Serialized trained XGBoost model

---

## 9. Experimental Results

### Test Set Performance

| Metric | Value |
|------|------|
| Mean Squared Error (MSE) | 0.00317 |
| Mean Absolute Error (MAE) | 0.0437 |
| R² Score | 0.8852 |

The high **R² score (0.8852)** indicates that the model explains a substantial proportion of the variance in network security risk, demonstrating strong generalization performance on unseen data.

---

## 10. Training Pipeline Orchestration

**File:** `pipeline/training_pipeline.py`

### Description

The training pipeline orchestrates all stages of the machine learning workflow in a **deterministic and reproducible** manner. Each stage produces explicit artifacts, enabling traceability and modular experimentation.

### Pipeline Stages

1. Data Ingestion  
2. Data Validation  
3. Data Transformation  
4. Model Training  

All intermediate outputs are **logged, versioned, and persisted** for reproducibility and debugging.

---

## 11. MongoDB Integration

MongoDB is accessed using **environment-based configuration**, ensuring security and deployment flexibility.

```python
import pymongo, certifi

ca = certifi.where()
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
collection = client[DB_NAME][COLLECTION_NAME]
```

This approach ensures:

- **Secure TLS-based communication** between the application and the database  
- **Decoupling of credentials from source code** via environment variables  
- **Easy adaptation across development and production environments**

---

## 12. Logging & Exception Handling

### Logging

A centralized logging mechanism captures detailed execution traces for each pipeline stage. This supports:

- Systematic debugging  
- Runtime monitoring  
- Experiment auditability and reproducibility  

### Exception Handling

All critical operations are wrapped in `try/except` blocks and raise a custom exception:

- `NetworkSecurityException`

This design ensures **consistent error propagation**, improved debuggability, and clean failure handling across the entire pipeline.

---

## 13. Streamlit Deployment

**File:** `app.py`

### Functionality

- End-to-end model training via a graphical user interface  
- CSV-based batch inference for offline evaluation  
- Automatic persistence of prediction results  

### Inference Workflow

```python
preprocessor = load_object("final_model/preprocessor.pkl")
model = load_object("final_model/model.pkl")

network_model = NetworkModel(
    preprocessor=preprocessor,
    model=model
)
````
predictions = network_model.predict(input_df)
Predictions are automatically stored for downstream analysis and evaluation.

## 14. Limitations & Future Work

While the current system demonstrates strong empirical performance, several research-oriented extensions are planned:

- **Incorporation of temporal sequence models** (LSTM, Temporal CNNs, Transformers) to capture sequential patterns in network traffic  
- **Cost-sensitive learning** to minimize false negatives in security-critical scenarios  
- **Adversarial robustness evaluation** against evasion and poisoning attacks  
- **Formal statistical significance testing** across model variants to ensure robust conclusions  
- **Fairness and bias analysis** in network security predictions to ensure equitable performance  

These directions align the project with ongoing research challenges in **applied machine learning for cybersecurity**.

