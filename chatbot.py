from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from knowledge_base import faq_data, faq_answers
from preprocess import preprocess

# Preprocess all questions
processed_questions = [preprocess(q) for q in faq_data]

# Convert to vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_questions)

def chatbot():
    print("🤖 Chatbot: Ask me anything about AI/ML/Python (type 'exit' to quit)")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        
        processed_input = preprocess(user_input)
        user_vec = vectorizer.transform([processed_input])
        
        similarity = cosine_similarity(user_vec, X)
        best_index = similarity.argmax()
        best_score = similarity[0][best_index]
        
        if best_score > 0.2:
            print("Chatbot:", faq_answers[best_index])
        else:
            print("Chatbot: Please ask related to AI, ML, or Python.")

if __name__ == "__main__":
    chatbot()