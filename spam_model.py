# Import libraries
import pandas as pd
import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="dinesh sharma",  # change this
    database="spam_detection"
)

cursor = conn.cursor()

# Load dataset
data = pd.read_csv(r"D:\SPAM EMAIL DETECTION\spam.csv", encoding='latin-1')

# Keep only needed columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels to numbers
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split data
X = data['message']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Convert text into numbers
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Test custom message
while True:
    msg = input("\nEnter a message (or type 'exit'): ")
    if msg.lower() == 'exit':
        break

    msg_vec = vectorizer.transform([msg])
    prediction = model.predict(msg_vec)

    if prediction[0] == 1:
        print("Spam Message 🚨")
    else:
        print("Not Spam ✅")

 # Save to MySQL
    sql = "INSERT INTO messages (message, prediction) VALUES (%s, %s)"
    cursor.execute(sql, (msg, result))
    conn.commit()

    print("Saved to MySQL ✅")

cursor.close()
conn.close()

