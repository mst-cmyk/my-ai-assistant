import streamlit as st
import google.generativeai as genai
import os
from PyPDF2 import PdfReader

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

st.set_page_config(page_title="AI PM Assessor", page_icon="💼")
st.title("💼 AI Business Value Assessor")
st.write("Native Engine Powered by Gemini 1.5 Flash")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Analyzing..."):
        try:
            reader = PdfReader(uploaded_file)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text()
            
            if not full_text.strip():
                st.warning("⚠️ This PDF seems empty or is a scanned image.")
            else:
                model = genai.GenerativeModel('gemini-1.5-flash')
                st.success("Document loaded successfully!")
                
                user_question = st.text_input("Ask the AI Product Director a question:")
                
                if user_question:
                    prompt = f"""
                    You are a Senior AI Product Director. 
                    Based ONLY on the following context, answer the question.
                    
                    Context:
                    {full_text[:30000]}
                    
                    Question: {user_question}
                    """
                    
                    response = model.generate_content(prompt)
                    st.write("### 💼 Evaluation Report:")
                    st.write(response.text)
                    
        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")
