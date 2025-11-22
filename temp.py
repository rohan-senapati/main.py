import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define folder path
base_dir = os.getcwd()
model_dir = os.path.join(base_dir, "model")
os.makedirs(model_dir, exist_ok=True)

# Example dataset (you can replace this with your own)
texts = [
    "AI is transforming the world",
    "Machine learning helps in automation",
    "Deep learning is part of AI"
]

# 1️⃣ Create and fit a CountVectorizer
cv = CountVectorizer(stop_words="english")
count_matrix = cv.fit_transform(texts)

# 2️⃣ Compute cosine similarity
cosine_sim = cosine_similarity(count_matrix)

# 3️⃣ Save both files in the 'model' folder
np.save(os.path.join(model_dir, "cosine_similarity.npy"), cosine_sim)
with open(os.path.join(model_dir, "count_vectorizer.pkl"), "wb") as f:
    pickle.dump(cv, f)

print("✅ Files successfully created inside:", model_dir)
