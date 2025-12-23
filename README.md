# Network Security in AI

🔗 **Live Deployment**  
👉 https://nagane09-network-security-in-ai-app-jbhxfu.streamlit.app/

---

# 🛡️ NetworkSecurity ML Project

This project implements a **Machine Learning pipeline for network security detection**. It fetches network data from MongoDB, performs preprocessing, trains ML models, and provides **real-time predictions via a Streamlit web application**.

---

## 🔹 Project Overview

- **Objective:** Predict network security issues based on historical network logs.
- **Approach:** Build a robust ML pipeline with data ingestion, validation, transformation, model training, and real-time inference.
- **Pipeline Stages:**
  1. Data Ingestion
  2. Data Validation
  3. Data Transformation
  4. Model Training
  5. Streamlit Deployment for prediction

- **Tech Stack:** Python, Pandas, NumPy, Scikit-learn, XGBoost, MongoDB, Streamlit, Joblib
- **Design Patterns:** Modular pipeline, artifact-based workflow, exception handling, and logging.


---

## 1️⃣ Data Ingestion

**File:** `components/data_ingestion.py`  
**Purpose:** Fetch data from MongoDB, clean, and store in local CSV. Split dataset into train/test.

**Technical Details:**
- Connects to MongoDB using `pymongo` with SSL verification (`certifi`).
- Reads collection into a Pandas DataFrame.
- Drops `_id` column and replaces `"na"` with `np.nan`.
- Stores raw data in **feature store**.
- Splits dataset using `train_test_split` from Scikit-learn.
- **Logging:** Records steps like fetching, cleaning, storing, and splitting.
- **Exception Handling:** Raises `NetworkSecurityException` on any failure.

**Artifact:** `DataIngestionArtifact` containing train/test CSV paths.

---

## 2️⃣ Data Validation

**File:** `components/data_validation.py`  
**Purpose:** Ensure the data quality and schema correctness before ML.

**Technical Details:**
- Validates required columns, target presence, and missing values.
- Detects schema mismatches and invalid entries.
- Produces cleaned train/test files ready for transformation.
- Logs all validation steps.
- Handles errors with `NetworkSecurityException`.

**Artifact:** `DataValidationArtifact` with validated train/test file paths.

---

## 3️⃣ Data Transformation

**File:** `components/data_transformation.py`  
**Purpose:** Preprocess input features for ML training.

**Technical Details:**
- **Feature Selection:** Drops target column from input features.
- **Target Cleaning:** Replaces `-1` in target column with `0` (binary label correction).
- **Imputation:** Uses `KNNImputer` (with `n_neighbors=3`) to fill missing values.
- **Pipeline:** Preprocessing pipeline using `Pipeline` from Scikit-learn.
- **Scaling/Normalization:** Optional addition of `StandardScaler` or other transformers.
- **Transformation:** Applies pipeline to train and test datasets.
- **Concatenation:** Combines transformed features with target labels for model consumption.
- **Persistence:** Saves transformed data as `.npy` arrays, preprocessor as `.pkl`.

**Artifact:** `DataTransformationArtifact` with paths to:
- Transformed train/test arrays
- Preprocessor object (`preprocessor.pkl`)

---

## 4️⃣ Model Training

**File:** `components/model_trainer.py`  
**Purpose:** Train ML models and generate predictions.

**Technical Details:**
- Models Tested: Random Forest, Gradient Boosting, XGBoost
- Final Selected Model: **XGBoost Regressor (`XGBRegressor`)**
- Hyperparameters: Tuned manually or via grid search
- Features used: All preprocessed network features
- Evaluation Metrics:
  - **MSE:** Mean Squared Error
  - **MAE:** Mean Absolute Error
  - **R² Score:** Model performance
- Model persistence: Saves trained model as `model.pkl`
- Logs training progress, evaluation metrics, and errors.

**Artifact:** `ModelTrainerArtifact` containing trained model path and metrics.

---

## 🔹 Model Training

- **Models Tested:**
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - **XGBoost Regressor (Selected)**

- **Final Model:** `XGBRegressor`  
  **Hyperparameters:**
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
  `````

------

## 📊 Model Validation Metrics (Test Set)

- **Mean Squared Error (MSE):** 0.00317  
- **Mean Absolute Error (MAE):** 0.0437  
- **R² Score:** 0.8852  

---

## 💾 Artifacts Saved

- `model.pkl` → Trained XGBoost model  
- `preprocessor.pkl` → Data transformation pipeline (KNNImputer + preprocessing)


------
## 5️⃣ Training Pipeline Orchestration

**File:** `pipeline/training_pipeline.py`  
**Purpose:** Centralized orchestration of all pipeline steps.

**Technical Details:**
- Initializes `TrainingPipelineConfig`.
- Executes stages sequentially:
  1. Data Ingestion
  2. Data Validation
  3. Data Transformation
  4. Model Training
- Returns `ModelTrainerArtifact` at the end.
- Logging records start/end of each stage.
- All exceptions are wrapped in `NetworkSecurityException`.

---

## 6️⃣ MongoDB Integration

- Connection using `pymongo` with environment variable `MONGODB_URL_KEY`.
- Example:

```python
import pymongo, certifi
ca = certifi.where()
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
collection = client[DB_NAME][COLLECTION_NAME]
```

----

## 7️⃣ Logging and Exception Handling

- **Logging:**  
  Every step of data ingestion, validation, transformation, and model training is logged using `logger.py` for traceability and debugging.

- **Exception Handling:**  
  Critical functions are wrapped in `try/except` blocks, raising `NetworkSecurityException` to ensure proper error tracking and debugging.

---

## 8️⃣ Streamlit Deployment

- **File:** `app.py`  
- **Purpose:** Provides a user interface for model training and real-time predictions.

### Features:

1. **Train Model:**  
   Runs the full pipeline from data ingestion to model training.

2. **Predict:**  
   - Upload CSV → Preprocess → Predict → Save results.  
   - Predictions are saved in `prediction_output/output.csv`.

### Workflow Example:

```python
preprocessor = load_object("final_model/preprocessor.pkl")
model = load_object("final_model/model.pkl")
network_model = NetworkModel(preprocessor=preprocessor, model=model)
y_pred = network_model.predict(input_df)
````
