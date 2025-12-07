import sys
import os
import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGODB_URL_KEY")
print(mongo_db_url)

import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline

import pandas as pd
import streamlit as st

from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

# MongoDB connection (kept as is)
client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
from networksecurity.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from networksecurity.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

# Streamlit App
st.title("Network Security ML App")

menu = ["Train Model", "Predict"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Train Model":
    st.header("Train the Network Security Model")
    if st.button("Run Training Pipeline"):
        try:
            train_pipeline = TrainingPipeline()
            train_pipeline.run_pipeline()
            st.success("Training completed successfully!")
        except Exception as e:
            st.error(f"Error: {e}")
            raise NetworkSecurityException(e, sys)

elif choice == "Predict":
    st.header("Upload CSV for Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Input Data:")
            st.dataframe(df.head())

            # Load preprocessor and model (kept as is)
            preprocessor = load_object("final_model/preprocessor.pkl")
            final_model = load_object("final_model/model.pkl")
            network_model = NetworkModel(preprocessor=preprocessor, model=final_model)

            y_pred = network_model.predict(df)
            df['predicted_column'] = y_pred

            st.write("Predictions:")
            st.dataframe(df)

            # Save output if needed
            os.makedirs("prediction_output", exist_ok=True)
            df.to_csv('prediction_output/output.csv', index=False)
            st.success("Predictions saved to prediction_output/output.csv")

        except Exception as e:
            st.error(f"Error: {e}")
            raise NetworkSecurityException(e, sys)
