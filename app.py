import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("dataset.csv", encoding='latin-1')
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

def predict_message(msg):
    msg_vec = vectorizer.transform([msg])
    return model.predict(msg_vec)[0]
#Streamlit
st.title("Spam Email Classifier")
st.write("Enter a message to check whether it is Spam or Not")
message = st.text_area("Enter Message")
if st.button("Check"):
    if message:
        result = predict_message(message)
        if result == "spam":
            st.error("SPAM MESSAGE!")
        else:
            st.success("NOT a Spam Message")
    else:
        st.warning("Please enter a message!")
st.sidebar.write("Model Accuracy")
st.sidebar.write(model.score(X_test, y_test))
