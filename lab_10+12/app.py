from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

df = pd.read_csv("cleaned_books.csv")
embeddings = np.load("book_embeddings.npy")
index = faiss.read_index("faiss_index.index")

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def search_books(query, count=5):
    query = clean_text(query)
    query_embedding = model.encode([query])

    distance, indices = index.search(query_embedding, count)

    results = []

    for i in range(count):
        idx = indices[0][i]

        results.append({
            "title": df['title'].iloc[idx],
            "author": df['authors'].iloc[idx]
        })

    return results

@app.route("/", methods=["GET", "POST"])
def home():
    results = []

    if request.method == "POST":
        query = request.form["query"]
        results = search_books(query)

    return render_template("index.html", results=results)
def search_books(query, count=5):
    query = clean_text(query)
    query_embedding = model.encode([query])

    distance, indices = index.search(query_embedding, count)

    results = []

    for i in range(count):
        idx = indices[0][i]

        results.append({
            "title": df['title'].iloc[idx],
            "author": df['authors'].iloc[idx],
            "rating": df['average_rating'].iloc[idx]
        })

    return results

if __name__ == "__main__":
    app.run(debug=True)
