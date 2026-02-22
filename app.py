from google import genai
import streamlit as st
from PyPDF2 import PdfReader


client = genai.Client(
    api_key=st.secrets["GOOGLE_API_KEY"],
    http_options={'api_version': 'v1'}
)

st.set_page_config(page_title="AI PM Assessor", page_icon="🎯")
st.title("🎯 AI Business Value Assessor")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    full_text = "".join([p.extract_text() for p in reader.pages if p.extract_text()])

    if full_text.strip():
        st.success("✅ AI Director is ready.")
        user_question = st.text_input("Question:")

        if user_question:
            with st.spinner("Analyzing..."):
                try:
                    response = client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=f"Context from PDF:\n{full_text[:12000]}\n\nQuestion: {user_question}"
                )
            
                    st.markdown("### 💼 Evaluation Report")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Final Debug Error: {e}")
