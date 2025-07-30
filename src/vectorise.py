import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import joblib
import os

os.makedirs(os.path.dirname("data/processed/feature_vector.joblib"), exist_ok=True)

df = pd.read_csv("data/processed/aita_processed.csv")

titles = df["title"].tolist()
bodies = df["body"].tolist()

vectoriser = TfidfVectorizer()
X_title = vectoriser.fit_transform(titles)
X_body = vectoriser.fit_transform(bodies)

X = hstack([X_title, X_body])
joblib.dump(X, "data/processed/feature_vector.joblib")
print("Preprocessing complete. Processed data saved to data/processed/aita_processed.csv")