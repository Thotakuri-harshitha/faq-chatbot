import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    tokens = nltk.word_tokenize(text)
    
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)