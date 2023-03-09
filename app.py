import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    
    for i in text:
        if i.isalnum():
            y.append(i)
    b = []      
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            b.append(i)
            
    c = []
    for i in b:
        c.append(ps.stem(i))
            
    return " ".join(c)

cv = pickle.load(open("vectorizerr.pkl",'rb'))
model = pickle.load(open('modell.pkl','rb'))

st.title("Sentiment Analysis")

input_sms = st.text_area("How was your experience?")

if st.button('Predict'):

    transformed_sms = transform_text(input_sms)

    vector_input = cv.transform([transformed_sms])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Positive Review")
    else:
        st.header("Negative Review")