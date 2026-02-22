import streamlit as st
import google.generativeai as genai
import os
from PyPDF2 import PdfReader

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

st.set_page_config(page_title="AI PM Assessor", page_icon="💼")
st.title("💼 AI Business Value Assessor")
st.write("Using Official Google AI SDK - Stable Mode")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Analyzing..."):
        try:
            reader = PdfReader(uploaded_file)
            full_text = "".join([page.extract_text() for page in reader.pages])
            
            if not full_text.strip():
                st.warning("⚠️ This PDF seems empty or is a scanned image.")
            else:
                model_name = 'models/gemini-1.5-flash'
                model = genai.GenerativeModel(model_name)
                
                st.success(f"Model {model_name} initialized successfully!")
                
                user_question = st.text_input("Ask the AI Product Director a question:")
                
                if user_question:
                    context_limit = 20000 
                    prompt = f"Context:\n{full_text[:context_limit]}\n\nQuestion: {user_question}"
                    
                    response = model.generate_content(prompt)
                    
                    st.write("### 💼 Evaluation Report:")
                    st.write(response.text)
                    
        except Exception as e:
            st.error(f"⚠️ API Error: {e}")
