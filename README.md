🌟 🩺 MULTI-DISEASE PREDICTION SYSTEM
Early Detection Powered by Machine Learning & Streamlit

📘 Description

The Multi-Disease Prediction App is a Streamlit-based Machine Learning application designed to assist in early detection of multiple diseases.
It predicts the likelihood of Kidney Disease, Liver Disease, and Parkinson’s Disease using trained ML models built with XGBoost, Logistic Regression, and other algorithms.

This application provides fast predictions, risk-level insights, and a user-friendly interface, making it suitable for healthcare learners, practitioners, and data science projects.
🌟 Key Features

🔍 Multi-Disease Support
Predicts:

Kidney Disease

Liver Disease

Parkinson’s Disease

⚡ Real-Time Predictions
Instant output with probability scores and risk level indicators.

🎨 Clean Streamlit UI
Easy-to-use interface for entering patient details and viewing results.

📈 Model Confidence & Visualization
Includes probability outputs and classification risk (Low / Medium / High).

🔐 Privacy-Friendly
No data stored — everything runs locally on user machine.

🧩 Modular ML Architecture
Each disease uses separate, optimized ML pipelines with preprocessing.

🏗️ Project Structure
📁 Multi-Disease-Prediction-App
│── app.py                     # Streamlit frontend
│── requirements.txt           # Python dependencies
│── README.md                  # Project documentation
│
├── models/                    # Saved ML models (XGBoost pipelines)
│     ├── kidney_xgb_pipeline.pkl
│     ├── liver_xgb_pipeline.pkl
│     ├── parkinson_xgb_pipeline.pkl
│
├── features/                  # Feature engineering artifacts
│     ├── kidney_features.pkl
│     ├── liver_features.pkl
│     ├── parkinson_features.pkl


▶️ How to Run the App
1. Install Dependencies
pip install -r requirements.txt

2. Run the Streamlit Application
streamlit run app.py

3. View in Browser
http://localhost:8501

🧠 Machine Learning Models Used

XGBoost Classifier

Logistic Regression

Random Forest (optional)

Robust scaling & preprocessing

Pickled pipelines for smooth deployment

🚀 Future Enhancements

Add more diseases (Diabetes, Heart Disease, etc.)

Add model explainability (SHAP charts)

Deploy on Streamlit Cloud or AWS

Add patient report PDF export
