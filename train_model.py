import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
import pickle
import warnings
import numpy as np
import os

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")

# Optional: Fix Loky CPU warning
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())

print("📥 Loading datasets...")
true = pd.read_csv("True.csv", encoding='utf-8')
fake = pd.read_csv("Fake.csv", encoding='utf-8')

true["label"] = "REAL"
fake["label"] = "FAKE"

data = pd.concat([true, fake], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
data['full_text'] = data['title'] + " " + data['text']
data = data[['full_text', 'label']].dropna()

X = data['full_text']
y = data['label']

print("🔤 Vectorizing text...")
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

print("🧠 Training advanced ensemble...")

# Base models with clean parameters
logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
xgb = XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)  # cleaned param
lgbm = lgb.LGBMClassifier(
    n_estimators=200,
    num_leaves=31,
    min_child_samples=5,
    random_state=42
)

# Weighted soft-voting ensemble
ensemble_model = VotingClassifier(
    estimators=[('lr', logreg), ('rf', rf), ('xgb', xgb), ('lgbm', lgbm)],
    voting='soft',
    weights=[2, 1, 2, 2]  # stronger models weighted more
)

# Calibrate probabilities for better confidence
ensemble_model = CalibratedClassifierCV(ensemble_model, cv=3)
ensemble_model.fit(X_train, y_train)

accuracy = ensemble_model.score(X_test, y_test)
print(f"✅ Ensemble model trained with accuracy: {accuracy:.2f}")

# Save model and vectorizer
pickle.dump(ensemble_model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("💾 Model and vectorizer saved successfully!")
