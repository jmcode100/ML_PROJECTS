import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load your pre-trained model and vectorizer
model = joblib.load(r"C:\Users\jiten\Desktop\Projectml2\Restaurant_review_model")
vectorizer = joblib.load(r"C:\Users\jiten\Desktop\Projectml2\count_v_res")

def preprocess_text(text):
    custom_stopwords = {'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't",
                         'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
                         'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
                         'needn', "needn't", 'shan', "shan't", 'no', 'nor', 'not', 'shouldn', "shouldn't",
                         'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english")) - custom_stopwords

    # Clean and preprocess text
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    review = " ".join(review)
    return review

def main():
    st.title("Restaurant Review Classification App")

    # Text input from the user
    user_input = st.text_area("Enter your restaurant review:", height=200)

    if st.button("Classify"):
        if user_input:
            processed_input = preprocess_text(user_input)
            processed_input_vectorized = vectorizer.transform([processed_input])
            prediction = model.predict(processed_input_vectorized)[0]
            sentiment = "Positive" if prediction == 1 else "Negative"
            st.success(f"Predicted Sentiment: {sentiment}")
        else:
            st.error("Please enter a review before clicking 'Classify'.")

if __name__ == "__main__":
    main()
