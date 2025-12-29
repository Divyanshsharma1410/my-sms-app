import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Divyansh's Portfolio", page_icon="👨‍💻", layout="wide")

# --- DOWNLOAD NLTK PACKAGES (Required for Streamlit Cloud) ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Projects", "Contact Me"])

# --- PAGE 1: HOME / ABOUT ME ---
if page == "Home":
    st.title("Hi, I'm Divyansh! 👋")
    st.subheader("B.Tech CSE (AI/ML) Student | Shri Vishwakarma Skill University")
    st.write("---")
    st.write("""
    ### 🚀 About Me
    I am a final-year Computer Science student specializing in **Artificial Intelligence and Machine Learning**. 
    I am passionate about building intelligent systems that solve real-world problems.
    
    **💡 Technical Skills:**
    * **Languages:** Python, C++
    * **Frameworks:** Streamlit, Flask
    * **Machine Learning:** Scikit-learn, Pandas, NumPy, NLTK
    * **Tools:** VS Code, Git, GitHub
    """)

# --- PAGE 2: PROJECTS (Your SMS App) ---
elif page == "Projects":
    st.title("💻 My Projects")
    
    tab1, tab2 = st.tabs(["📩 SMS Spam Classifier", "Coming Soon"])
    
    with tab1:
        st.header("SMS Spam Detection System")
        st.write("This AI-powered tool classifies messages as 'Spam' or 'Not Spam'.")

        # --- LOAD SAVED MODELS ---
        # Ensure 'vectorizer.pkl' and 'model.pkl' are in your GitHub repo
        try:
            tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
            model = pickle.load(open('model.pkl', 'rb'))
        except FileNotFoundError:
            st.error("⚠️ Error: Model files not found. Please upload vectorizer.pkl and model.pkl to GitHub.")
            st.stop()

        # --- PREPROCESSING FUNCTION ---
        ps = PorterStemmer()

        def transform_text(text):
            text = text.lower()
            text = nltk.word_tokenize(text)
            y = []
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

        # --- USER INTERFACE ---
        input_sms = st.text_area("Enter the message to check:", height=150)

        if st.button('Analyze Message'):
            if input_sms:
                transformed_sms = transform_text(input_sms)
                vector_input = tfidf.transform([transformed_sms])
                result = model.predict(vector_input)[0]

                if result == 1:
                    st.error("🚨 This message is **SPAM**")
                else:
                    st.success("✅ This message is **NOT SPAM**")
            else:
                st.warning("Please enter a message first.")

    with tab2:
        st.info("More projects (like 'Kisan Mitra') coming soon!")

# --- PAGE 3: CONTACT ---
elif page == "Contact Me":
    st.title("📬 Get in Touch")
    st.write("I am open to internship opportunities and collaborations.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**📧 Email:**")
        st.write("sharmadivyansh1410@gmail.com")
    with col2:
        st.write("**🐙 GitHub:**")
        st.write("[github.com/Divyanshsharma1410](https://github.com/Divyanshsharma1410)")