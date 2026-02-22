
from google import genai
import streamlit as st
import os
from PyPDF2 import PdfReader

client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

st.title("💼 AI Business Value Assessor")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)

    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text

    if full_text.strip():
        st.success("AI Director is ready to analyze.")

        user_question = st.text_input("Ask a question:")

        if user_question:
            with st.spinner("Thinking..."):
                response = client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=f"""
                    Context from PDF:
                    {full_text[:20000]}

                    Question:
                    {user_question}
                    """
                )

                st.write("### 💼 Evaluation Report:")
                st.write(response.text)

from google import genai

client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

models = client.models.list()

for m in models:
    st.write(m.name)
