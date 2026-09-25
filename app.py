import streamlit as st
import joblib

# Load trained model and TF-IDF vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("📧 Spam Email Detection")
st.write("Enter an email below to check whether it is Spam or Ham.")

email = st.text_area("Enter Email Text")

if st.button("Predict"):
    if not email.strip():
        st.warning("Please enter an email.")
    else:
        email_tfidf = vectorizer.transform([email])
        prediction = model.predict(email_tfidf)[0]

        if prediction == 1:
            st.error("🚨 SPAM EMAIL")
        else:
            st.success("✅ HAM (Not Spam)")
