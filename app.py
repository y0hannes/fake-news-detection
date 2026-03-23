import streamlit as st
import os

# Page configuration
st.set_page_config(
    page_title="TruthScanner AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
