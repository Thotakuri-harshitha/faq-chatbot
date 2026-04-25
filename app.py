import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from knowledge_base import faq_data, faq_answers
from preprocess import preprocess

# Preprocess
processed_questions = [preprocess(q) for q in faq_data]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_questions)

st.title("🤖 AI Chatbot (Python / AI / ML)")

user_input = st.text_input("Ask your question:")

if user_input:
    processed_input = preprocess(user_input)
    user_vec = vectorizer.transform([processed_input])
    
    similarity = cosine_similarity(user_vec, X)
    best_index = similarity.argmax()
    best_score = similarity[0][best_index]
    
    if best_score > 0.2:
        st.success(faq_answers[best_index])
    else:
        st.warning("Please ask related to AI, ML, or Python.")