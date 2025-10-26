from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html', prediction_text="", color="#000")

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news']
    
    if not news_text.strip():
        return render_template('index.html', prediction_text="⚠️ Please enter some news content.", color="#000")

    # Transform input
    input_vec = vectorizer.transform([news_text])
    prediction = model.predict(input_vec)[0]
    confidence = np.max(model.predict_proba(input_vec)) * 100

    # Set color and message
    if prediction == "FAKE":
        result = f"🚨 This news appears to be FAKE (Confidence: {confidence:.2f}%)"
        color = "#e74c3c"  # red
    else:
        result = f"✅ This news appears to be REAL (Confidence: {confidence:.2f}%)"
        color = "#27ae60"  # green

    return render_template('index.html', prediction_text=result, color=color)

if __name__ == "__main__":
    app.run(debug=True)
