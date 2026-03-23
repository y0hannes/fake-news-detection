import streamlit as st
import os

# Page configuration
st.set_page_config(
    page_title="TruthScanner AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

def main():
    st.markdown("# 🔍 TruthScanner AI")
    st.markdown("### Advanced Misinformation Detection System")
    
    # Simple sidebar
    with st.sidebar:
        st.header("About")
        st.info("TruthScanner AI uses a Deep Neural Network to analyze linguistic patterns and predict the authenticity of news headlines.")

    # Main input area
    st.write("---")
    news_input = st.text_area(
        "Paste the news headline or content here:",
        height=200,
        placeholder="Enter the news text you want to verify..."
    )

    if st.button("Analyze Authenticity", type="primary"):
        if news_input.strip():
            st.warning("Analysis logic integration pending...")
        else:
            st.error("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
