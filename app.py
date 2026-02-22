import streamlit as st
import google.generativeai as genai
import os
from PyPDF2 import PdfReader

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=os.environ["GOOGLE_API_KEY"], transport='rest')

st.set_page_config(page_title="AI PM Assessor", page_icon="💼")
st.title("💼 AI Business Value Assessor")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:
    try:
        reader = PdfReader(uploaded_file)
        full_text = "".join([page.extract_text() for page in reader.pages])
        
        if full_text.strip():
            model = genai.GenerativeModel('models/gemini-1.5-flash')
            st.success("AI Director is ready to analyze.")
            
            user_question = st.text_input("Ask a business question:")
            if user_question:
                with st.spinner("Analyzing via Stable V1 API..."):
                    prompt = f"Context:\n{full_text[:20000]}\n\nQuestion: {user_question}"
                    response = model.generate_content(prompt)
                    if response.text:
                        st.write("### 💼 Evaluation Report:")
                        st.write(response.text)
        else:
            st.warning("PDF appears empty.")
    except Exception as e:
        st.error(f"Execution Error: {e}")
