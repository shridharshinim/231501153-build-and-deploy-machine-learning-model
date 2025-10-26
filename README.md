📰 Advanced Fake News Detection System
Project Overview 🌟

The rapid spread of misinformation on social media has created an urgent need for automated systems to reliably detect fake content.

This project presents a TF-IDF + Ensemble-based system to classify news articles as REAL or FAKE.

Key Features:

✅ TF-IDF text encoding for contextual understanding.

🔹 Ensemble Learning: Logistic Regression, Random Forest, XGBoost, LightGBM.

📊 Calibrated confidence scores for prediction reliability.

🌐 Flask deployment with a color-coded interactive frontend.

Experiments Covered 🔬

Experiment 2: Supervised Learning Models

TF-IDF vectorization

Ensemble classifiers: Logistic Regression, Random Forest, XGBoost, LightGBM

Confidence calibration using CalibratedClassifierCV

Experiment 9: REST API for Model Deployment

Flask web application for real-time predictions

Interactive color-coded results:

✅ REAL news (green)

🚨 FAKE news (red)

Folder Structure 📂
FakeNewsProject/
├─ app.py                  # Flask application
├─ train_model.py          # Model training script
├─ model.pkl               # Trained ensemble model
├─ vectorizer.pkl          # TF-IDF vectorizer
└─ templates/
    └─ index.html          # Frontend HTML template

Setup & Installation ⚡

Clone the project or download the zip.

Install dependencies:

pip install pandas numpy scikit-learn xgboost lightgbm flask


Train the model:

python train_model.py


Run the Flask app:

python app.py


Open in browser:

http://127.0.0.1:5000/

Usage 🖱️

Paste a news headline or full article into the textarea.

Click Check Authenticity.

View prediction:

✅ REAL news (green)

🚨 FAKE news (red)

Confidence score is displayed alongside the result.

Sample Real News for Testing ✅

“NASA’s Artemis I mission successfully completes moon flyby.”

“OpenAI releases GPT-5 API for developers.”

“WHO recommends new malaria vaccine for children in Africa.”

Sample Fake News for Testing 🚨

“Drinking 3 liters of lemon water cures cancer overnight!”

“Government to make watching movies illegal next month.”

“Scientists warn Earth will collide with Mars next week!”

Future Enhancements 🔧

Integrate Google Gemini / GPT embeddings for semantic analysis.

Add images and metadata for multimodal detection.

Containerize using Docker for cloud deployment.

Add SHAP/LIME explainability to highlight influential words in predictions.



References 📚

Kaggle Fake News Dataset

Scikit-learn Documentation

XGBoost Documentation

LightGBM Documentation
