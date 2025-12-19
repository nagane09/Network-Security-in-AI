# Network Security in AI

🔗 **Live Deployment**  
👉 https://nagane09-network-security-in-ai-app-jbhxfu.streamlit.app/

---

## 🔐 What This Project Is Trying to Do

**Network Security in AI** is an end-to-end **machine learning–based network intrusion detection system**.

The goal of this project is to:
* Detect **malicious network traffic**
* Classify whether incoming data is **normal or an attack**
* Provide **real-time predictions** through a web interface
* Log and store predictions for monitoring and analysis

This project demonstrates how **AI can be used to enhance cybersecurity** by automatically identifying threats instead of relying only on rule-based systems.

---

## 🧠 Problem Statement

Traditional network security systems:
* Depend heavily on static rules
* Struggle with new or evolving attack patterns
* Require constant manual updates

This project solves that by using **Machine Learning models** that learn patterns from historical network traffic data and generalize to unseen attacks.

---

## 📄 What Each Important File Does

### `app.py`
* Streamlit web application
* Takes user input (network data)
* Loads trained model
* Displays prediction results in real time

### `main.py`
* Orchestrates the full ML pipeline
* Triggers data ingestion, preprocessing, training, and evaluation

### `push_data.py`
* Pushes or ingests new data into the system
* Useful for batch or streaming-based predictions

### `test_mongo.py`
* Tests database connectivity
* Ensures predictions can be stored and retrieved correctly

### `final_model/`
* Contains the trained machine learning model files
* Used during inference

### `networksecurity/`
* Core backend logic
* Modularized components for:
  * Data ingestion
  * Data transformation
  * Model training
  * Prediction pipeline

---

## 🔄 End-to-End Project Pipeline (What You Added)

This is a **complete end-to-end AI system**, not just a model:

1. **Data Ingestion**
   * Network traffic data loaded from datasets
2. **Data Validation**
   * Schema validation ensures correct input format
3. **Data Preprocessing**
   * Cleaning, encoding, scaling features
4. **Model Training**
   * Machine learning model trained on network data
5. **Model Evaluation**
   * Performance evaluation before deployment
6. **Model Saving**
   * Best model stored in `final_model/`
7. **Prediction Pipeline**
   * New data passed through trained model
8. **Web Deployment**
   * Streamlit app for real-time predictions
9. **Logging & Monitoring**
   * Logs stored for debugging and analysis

---

## 🤖 Models Used

* **Supervised Machine Learning Models**
  * Trained on labeled network traffic data
* Models focus on:
  * Binary or multi-class intrusion detection
  * Normal vs malicious traffic classification

> The architecture allows **easy replacement or extension** with advanced models such as Random Forest, XGBoost, or Deep Learning models.

---

## 🛠️ Tech Stack

### **Programming Language**
* Python

### **Machine Learning**
* Scikit-learn
* NumPy
* Pandas

### **Web Application**
* Streamlit

### **Database**
* MongoDB (for storing predictions/logs)

### **DevOps & MLOps**
* Modular pipeline design
* Logging system
* Model versioning

---

## 🚀 Deployment

* Deployed using **Streamlit**
* Accessible via browser
* No complex setup required for end users


---

## 📊 What Makes This More Than Just an AI Project?

In addition to **AI models**, this project includes:

✅ End-to-end pipeline  
✅ Real-time deployment  
✅ Logging and monitoring  
✅ Data validation  
✅ Modular and scalable architecture  
✅ Production-ready structure  
✅ Database integration  

This makes it a **complete AI + Software Engineering project**, suitable for real-world use.

---

## 🔮 Future Enhancements

* Add deep learning–based intrusion detection
* Real-time packet capture integration
* Role-based dashboards
* Alert notifications for detected attacks
* Dockerization for easier deployment

---

## 🙌 Final Note

This project demonstrates how **Artificial Intelligence + Software Engineering + Deployment** can be combined to build a real-world **network security solution**.
---

