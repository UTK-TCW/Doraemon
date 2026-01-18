import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import joblib
import time
import os

NLTK_DATA_PATH = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(NLTK_DATA_PATH)

nltk.download("punkt", download_dir=NLTK_DATA_PATH)
nltk.download("punkt_tab", download_dir=NLTK_DATA_PATH)
# --- 1. Page Config ---
st.set_page_config(
    page_title="Melody Mind",
    page_icon="📜", # Changed icon to scroll/paper
    layout="wide"
)

# --- 2. Custom CSS ---
st.markdown("""
    <style>
/* ---------- HANDWRITTEN MARGINALIA ---------- */

/* Sidebar container feels like page margin */
section[data-testid="stSidebar"] {
    background: none;
    border-right: 1px solid #e2dbcf;
    padding: 4rem 1.5rem 2rem 1.2rem;
}

/* All sidebar text becomes handwritten */
section[data-testid="stSidebar"] * {
    font-family: "Bradley Hand", "Segoe Script", "Comic Sans MS", cursive;
    font-size: 0.85rem;
    color: #6a5f52;
    line-height: 1.6;
}

/* Sidebar title = note header */
section[data-testid="stSidebar"] h1 {
    font-size: 0.9rem;
    letter-spacing: 1px;
    text-transform: none;
    margin-bottom: 1.8rem;
    transform: rotate(-1deg);
}

/* Paragraphs feel casually written */
section[data-testid="stSidebar"] p {
    margin-bottom: 1.2rem;
    transform: rotate(-0.6deg);
}

/* Genre list looks like scribbled notes */
section[data-testid="stSidebar"] code {
    background: none;
    padding: 0;
    font-size: 0.75rem;
    color: #7a6e61;
}

/* Each genre slightly misaligned (imperfection illusion) */
section[data-testid="stSidebar"] code:nth-child(odd) {
    display: inline-block;
    transform: rotate(0.8deg);
}

section[data-testid="stSidebar"] code:nth-child(even) {
    display: inline-block;
    transform: rotate(-0.8deg);
}

/* Divider looks like a pencil stroke */
section[data-testid="stSidebar"] hr {
    border: none;
    height: 1px;
    background: #cfc6b8;
    margin: 1.5rem 0;
    opacity: 0.6;
}
</style>
    """, unsafe_allow_html=True)

# --- 3. Load Resources ---
@st.cache_resource
def load_models():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
    
    vec = joblib.load('./streamPlus/tfidf_vectorizer.joblib') 
    mod = joblib.load('./streamPlus/svm_model.joblib')
    return vec, mod

try:
    vectorizer, model = load_models()
    model_loaded = True
except FileNotFoundError:
    st.error("⚠️ Error: Model files not found! Make sure 'svm_model.joblib' and 'tfidf_vectorizer.joblib' are in the same folder.")
    model_loaded = False

genre = ['Rock' ,'Metal', 'Pop', 'Indie', 'R&B' ,'Folk' ,'Electronic', 'Jazz', 'Hip-Hop', 'Country']

# --- 4. Sidebar ---
with st.sidebar:
    st.title("📜 Melody Mind")
    st.write("An AI that reads between the lines.")
    st.markdown("---")
    st.write("**Supported Genres:**")
    st.markdown(", ".join([f"`{g}`" for g in genre]))

# --- 5. Main UI ---
st.markdown("<h1 style='text-align: center;'>🎵 Song Genre Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; max-width:520px; margin:0 auto 3rem;'>"
    "Paste the lyrics below to detect the musical vibe."
    "</p>",
    unsafe_allow_html=True
)

col1, col2 = st.columns([3, 2])

with col1:
    user_input = st.text_area("Paste Lyrics:", height=250, placeholder="Type lyrics here...")
    predict_btn = st.button("Analyze Text")

with col2:
    st.markdown("<br>", unsafe_allow_html=True) 
    if predict_btn and model_loaded:
        if user_input.strip() != "":
            with st.spinner('Analyzing ink...'):
                time.sleep(0.5) 
                
                # Preprocessing
                cleaned_text = user_input.replace('\n',' ').replace('.',' ').replace(',',' ').lower()
                tokens = word_tokenize(cleaned_text)
                stop_words = set(stopwords.words('english'))
                filtered_tokens = [w for w in tokens if w not in stop_words and w.isalnum()]
                stemmer = PorterStemmer()
                stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]

                if not stemmed_tokens:
                    st.warning("⚠️ Please enter valid lyrics.")
                else:
                    final_input = [" ".join(stemmed_tokens)] 
                    
                    # Prediction
                    vectors = vectorizer.transform(final_input)
                    prediction_idx = model.predict(vectors)[0]
                    
                    if isinstance(prediction_idx, str):
                        result_genre = prediction_idx
                    else:
                        idx = int(prediction_idx) // 100 
                        result_genre = genre[idx]

                    # Result Display
                    st.markdown(f"""
                        <div class="genre-card">
                            <h3>Vibe Detected</h3>
                            <h1 style="
    color:#9c2f23;
    font-size:3rem;
    font-family:'Courier New', monospace;
    letter-spacing:2px;
    margin-top:10px;
">{result_genre}</h1>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("The page is blank. Write something first!")
