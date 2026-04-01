import streamlit as st
import os
import numpy as np
from utils import load_artifacts
from predict import predict_news
from preprocess import clean_text

@st.cache_resource
def get_model_and_vectorizer():
    model_path = 'fake_news_model.keras'
    vec_path = 'vectorizer.joblib'
    if os.path.exists(model_path) and os.path.exists(vec_path):
        return load_artifacts(model_path, vec_path)
    return None, None

# Page configuration
st.set_page_config(
    page_title="TruthScanner AI | Fake News Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Meta Information for SEO (Manual injection for UI feel)
st.title("🛡️ TruthScanner AI")
st.caption("Empowering users with Neural Network-driven misinformation detection.")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stTextArea textarea {
        background-color: #1e2227;
        color: #ffffff;
        border-radius: 15px;
        border: 1px solid #3e4451;
    }
    .stButton>button {
        background: linear-gradient(45deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 10px 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .result-card {
        padding: 25px;
        border-radius: 20px;
        margin: 20px 0;
        border: 1px solid #3e4451;
        background: rgba(30, 34, 39, 0.6);
        backdrop-filter: blur(10px);
    }
    </style>
    """, unsafe_allow_html=True)

def get_explanation(text, vectorizer, model):
    # Simple keyword extraction based on TF-IDF features present in the text
    feature_names = vectorizer.get_feature_names_out()
    text_vec = vectorizer.transform([text]).toarray()[0]
    
    # Get top 5 words that contributed most to this specific vectorization
    top_indices = np.argsort(text_vec)[-5:][::-1]
    top_words = [feature_names[i] for i in top_indices if text_vec[i] > 0]
    
    return top_words

def main():
    model, vectorizer = get_model_and_vectorizer()
    
    st.markdown("# 🔍 TruthScanner AI")
    st.markdown("### Advanced Misinformation Detection System")
    
    # Sidebar
    with st.sidebar:
        st.header("App Stats")
        if model:
            st.success("Model Status: Online")
            st.write("Architecture: Feedforward NN")
            st.write("Vectorizer: TF-IDF (10k features)")
        else:
            st.error("Model Status: Offline")
            st.warning("Please train the model using main.py first.")
        
        st.write("---")
        st.header("📜 Analysis History")
        if 'history' not in st.session_state:
            st.session_state.history = []
        
        for item in st.session_state.history[-5:]:
            with st.expander(f"{item['label']} ({item['conf']:.1%})"):
                st.write(f"*{item['text'][:100]}...*")

    # Main input area
    st.write("---")
    news_input = st.text_area(
        "Paste the news headline or content here:",
        height=200,
        placeholder="Enter the news text you want to verify..."
    )

    if st.button("Analyze Authenticity", type="primary"):
        if not model or not vectorizer:
            st.error("System components not loaded. Ensure artifacts exist.")
            return

        if news_input.strip():
            with st.spinner("Analyzing linguistic patterns..."):
                # Prediction logic
                cleaned = clean_text(news_input)
                vec = vectorizer.transform([cleaned])
                prob = model.predict(vec, verbose=0)[0][0]
                
                label = "Fake News" if prob > 0.5 else "Real News"
                confidence = prob if prob > 0.5 else (1 - prob)
                color = "#ff4b4b" if label == "Fake News" else "#28a745"
                
                # Update history
                st.session_state.history.append({
                    'label': label,
                    'conf': confidence,
                    'text': news_input
                })
                
                st.markdown(f"""
                    <div class="result-card" style="border-left: 5px solid {color};">
                        <h2 style="color: {color}; margin-top: 0;">{label} Detected</h2>
                        <p>Our AI system is <b>{confidence*100:.2f}%</b> confident in this prediction.</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Visual Confidence Bar
                st.progress(float(confidence))
                
                # Explanation Section
                st.markdown("### 🧬 Analysis Breakdown")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Why this classification?**")
                    if label == "Fake News":
                        st.write("The AI detected patterns commonly associated with emotionally charged or hyperbolic misinformation.")
                    else:
                        st.write("The linguistic structure aligns with standard journalistic reporting and factual consistency.")
                
                with col2:
                    keywords = get_explanation(cleaned, vectorizer, model)
                    st.write("**Key Linguistic Markers:**")
                    st.write(", ".join([f"`{w}`" for w in keywords]) if keywords else "Internal neural patterns")
                
                # Download Report
                report = f"""TRUTHSCANNER AI - ANALYSIS REPORT
----------------------------------
News Input: {news_input}
Result: {label}
Confidence: {confidence*100:.2f}%
Keywords: {', '.join(keywords) if keywords else 'N/A'}
Date: {np.datetime64('now')}
"""
                st.download_button(
                    label="📥 Download Analysis Report",
                    data=report,
                    file_name="truthscanner_report.txt",
                    mime="text/plain",
                )

        else:
            st.error("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
