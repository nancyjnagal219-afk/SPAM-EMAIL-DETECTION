import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Title
st.title("📧 Spam Email Detection App")
st.write("This app detects whether a message is Spam or Not Spam.")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("D:\SPAM EMAIL DETECTION\spam.csv", encoding='latin-1')
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    return data

data = load_data()

# Split data
X = data['message']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# User input
user_input = st.text_area("Enter your message:")

if st.button("Check"):
    if user_input:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)

        if prediction[0] == 1:
            st.error("🚨 This is a Spam Message!")
        else:
            st.success("✅ This is Not Spam (Safe Message).")
    else:
        st.warning("Please enter a message first.")



 # dir
#  python -m streamlit run spam-email-detection\app.py