import streamlit as st
import joblib
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import stopwords
import string
import nltk
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)
    
# Load the pre-trained models
cv = joblib.load(open('vector.pkl', 'rb')) 
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app title
st.title("Phishing Site Detection")

# Text area for input
input_sms = st.text_area("Enter the message")

# Predict button
if st.button('Predict'):
    # Preprocess the input
    transformed_url = transform_text(input_sms)
    
    # Vectorize the input
    vector_input = cv.transform([transformed_url])
    
    # Predict using the loaded model
    result = model.predict(vector_input)[0]
    
    # Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
