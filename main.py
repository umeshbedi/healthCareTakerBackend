from flask import Flask, request
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.stem.porter import PorterStemmer

ds = pd.read_csv("https://raw.githubusercontent.com/umeshbedi/healthCareTakerBackend/refs/heads/main/Symptoms%20-%20Sheet1.csv")

# Initialize CountVectorizer with custom tokenizer and Hindi stop words
vectorizer = CountVectorizer(max_features=5000, stop_words="english")

# Fit and transform the data
vectors = vectorizer.fit_transform(ds['symptoms'])

# vectorizer.get_feature_names_out()


ps = PorterStemmer()

def stem(text):
  y = []
  for i in text.split():
    y.append(ps.stem(i))
  return " ".join(y)

ds['symptoms'] = ds['symptoms'].apply(stem)

app = Flask(__name__)

@app.route("/detect", methods=["GET"])
def detect():
    user_symptoms = request.args.get("symptoms")

    # Vectorize the user input
    user_vector = vectorizer.transform([user_symptoms])

    # Step 3: Compute cosine similarity between user symptoms and disease symptoms
    similarity = cosine_similarity(user_vector, vectors)

    # Step 4: Get the most related disease (highest cosine similarity score)
    most_similar_idx = similarity.argmax()
    most_similar_disease = ds['disease'][most_similar_idx]

    # Output the most similar disease
    print(f"The most related disease is: {most_similar_disease}")
    print(f"Tips for Prevention: {ds['preventions'][most_similar_idx]}")
    return f"""<h1>{most_similar_disease}</h1>
    <p>{ds['preventions'][most_similar_idx]}</p>
    """

