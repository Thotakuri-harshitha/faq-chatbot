import streamlit as st
import os
import nltk

# ✅ Fix NLTK for Streamlit Cloud
nltk_data_path = "/tmp/nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)

# ✅ Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from knowledge_base import faq_data, faq_answers
from preprocess import preprocess

# ✅ Preprocess questions
processed_questions = [preprocess(q) for q in faq_data]

# ✅ Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_questions)

# ✅ UI
st.title("🤖 AI Chatbot (Python / AI / ML)")

user_input = st.text_input("Ask your question:")

if user_input:
    processed_input = preprocess(user_input)
    user_vec = vectorizer.transform([processed_input])
    
    similarity = cosine_similarity(user_vec, X)
    best_index = similarity.argmax()
    best_score = similarity[0][best_index]
    
    # ✅ Improved response logic
    if best_score > 0.3:
        st.success(faq_answers[best_index])
    else:
        st.warning("❗ Please ask a question related to AI, ML, or Python.")
