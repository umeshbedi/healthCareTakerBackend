from flask import Flask, request, jsonify
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.stem.porter import PorterStemmer

# from googletrans import Translator
# translator = Translator()

from upload import upload_image

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
    
    # get_user_symptoms = request.args.get("symptoms")

    # detected_lang = translator.detect(get_user_symptoms)
    
    # print(detected_lang)


    user_symptoms = request.args.get("symptoms")
    
    # user_symptoms = translator.translate(get_user_symptoms, dest="en")

    

    # Vectorize the user input
    user_vector = vectorizer.transform([user_symptoms])

    # Step 3: Compute cosine similarity between user symptoms and disease symptoms
    similarity = cosine_similarity(user_vector, vectors)

    # Step 4: Get the most related disease (highest cosine similarity score)
    most_similar_idx = similarity.argmax()
    
    most_similar_disease = ds['disease'][most_similar_idx]

    highest_similarity_score = float(similarity[0, most_similar_idx])


    
    # print(most_similar_idx)
    # Output the most similar disease
    # print(f"The most related disease is: {most_similar_disease}")
    # print(f"Tips for Prevention: {ds['preventions'][most_similar_idx]}")

    if highest_similarity_score >= 0.1:
      return jsonify({
      "disease":most_similar_disease,
      "prevention":ds["preventions"][most_similar_idx],
      "status":"success",
      "accuracy":f"{(highest_similarity_score*100):.2f}"
    })
      
    else:
      return jsonify({
      "message":"Sorry! No disease found. Please provide me more than 2 or 3 symptoms.",
      "status":"not found"
    })
      

    


@app.route("/upload", methods=["POST"])
def uploading():
   return upload_image()

if __name__ == "__main__":
  app.run(host='0.0.0.0', port=5000, debug=True)
    

