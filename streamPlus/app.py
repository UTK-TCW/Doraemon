import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import joblib
# Download required resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

vectorizer = joblib.load('tfidf_vectorizer.joblib')
model = joblib.load('svm_model.joblib')
genre = ['Rock' ,'Metal', 'Pop', 'Indie', 'R&B' ,'Folk' ,'Electronic', 'Jazz', 'Hip-Hop',
 'Country']

st.title("🎵 Song Genre Predictor")
st.write("Enter lyrics below to see if its Genre!")

user_input = st.text_area("Paste Lyrics Here:", height=150)

if st.button("Predict Genre"):
    if user_input.strip() != "":

        cleaned_text = user_input.replace('\n',' ').replace('.',' ').replace(',',' ').replace('\'','').lower()
    # Word Splitting
        tokens = word_tokenize(cleaned_text)
      #print(f"Tokens: {tokens}")

      # Stop Word Removal
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [w for w in tokens if w not in stop_words and w.isalnum()]
      #print(f"After removing stop words: {filtered_tokens}")

      # Stemming
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
      #print(f"Stemmed Tokens: {stemmed_tokens}")
        print(stemmed_tokens)
        if not stemmed_tokens:
           st.warning("Dhang ka daal re.")
        else:
            vectors = vectorizer.transform(stemmed_tokens)
            
            # Predict
            prediction = model.predict(vectors)[0]
            i = prediction //100
            prediction = genre[i]
            
            # Show Result
            st.success(f"I predict this song is: **{prediction}**")
    else:
        st.warning("Please enter some text first.")