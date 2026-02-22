import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. Insert your API Key here (keep the quotes)
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(page_title="AI PM Resume Screener", page_icon="💼")
st.title("💼 AI Business Value Assessor")
st.write("Upload a resume or report. I will evaluate it strictly based on [Business Impact] and [Product Sense].")

# 2. Upload PDF
uploaded_file = st.file_uploader("Please upload a PDF file", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Analyzing from a business perspective..."):
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load_and_split()
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = Chroma.from_documents(documents=pages, embedding=embeddings)
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        
        # The Custom Prompt (The "Soul" of the AI PM)
        prompt_template = """
        You are an extremely strict Senior AI Product Director. When evaluating this document, you do not care about basic coding syntax. You only focus on the following three core dimensions:
        1. [Business Value]: Did the project save money, generate revenue, or significantly improve efficiency for the company? Are there specific data metrics (e.g., conversion rate, ROI) to support this?
        2. [0 to 1 Execution]: Is there evidence of leading cross-functional communication, requirement gathering, or driving a product from concept to launch?
        3. [Storytelling]: Does the document demonstrate the ability to translate complex technologies (e.g., data cleaning, model training) into clear business narratives?
        
        If the document is filled with technical jargon but lacks business thinking, sternly point out its "lack of product sense." If you see excellent potential for business translation, highlight it.

        Please answer the user's question based ONLY on the retrieved context below.
        Context: {context}
        
        User Question: {question}
        
        Your Evaluation:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, 
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": PROMPT} 
        )
        
        st.success("Analysis complete! I have put on my AI Product Director glasses.")
        
        # 3. Q&A Box
        user_question = st.text_input("Ask me a question (e.g., Evaluate the business potential of this candidate):")
        if user_question:
            response = qa_chain.invoke(user_question)
            st.write("### 💼 Evaluation Report:")
            st.write(response["result"])