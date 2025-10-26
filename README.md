📰 Advanced Fake News Detection System
Project Overview 🌟

The rapid spread of misinformation on social media has created an urgent need for automated systems that can reliably detect fake content.

This project presents a multimodal ensemble-based approach to classify social media posts as REAL or FAKE.

Key features:

TF-IDF text encoding for capturing contextual information.

Ensemble learning using Logistic Regression, Random Forest, XGBoost, LightGBM.

Calibrated confidence scores for reliable predictions.

Flask deployment with a color-coded frontend for real-time news classification.

Experiments Covered 🔬

This project primarily implements two experiments:

Experiment 2: Supervised Learning Models

Text classification using TF-IDF.

Ensemble methods: Logistic Regression, Random Forest, XGBoost, LightGBM.

Calibrated confidence using CalibratedClassifierCV.

Experiment 9: REST API for Model Deployment

Flask web app to serve the trained model.

Interactive frontend with color-coded results:

✅ Green → REAL news

🚨 Red → FAKE news

Ready for Docker deployment for portability and scalability.

Folder Structure 📂
FakeNewsProject/
├─ app.py                  # Flask application
├─train_model.py           # Script to train ensemble model
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

View prediction with confidence:

✅ REAL news (green)

🚨 FAKE news (red)

Sample Real News for Testing ✅

“NASA’s Artemis I mission successfully completes moon flyby.”

“OpenAI releases GPT-5 API for developers.”

“WHO recommends new malaria vaccine for children in Africa.”

Sample Fake News for Testing 🚨

“Drinking 3 liters of lemon water cures cancer overnight!”

“Government to make watching movies illegal next month.”

“Scientists warn Earth will collide with Mars next week!”

Future Enhancements 🔧

Integrate Google Gemini / GPT embeddings for semantic understanding.

Include image and metadata analysis for multimodal detection.

Deploy using Docker or cloud services for scalability.

Add SHAP/LIME explainability to highlight influential words.


References 📚

Kaggle Fake News Dataset

Scikit-learn Documentation

XGBoost Documentation

LightGBM Documentation
