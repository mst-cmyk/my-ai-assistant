import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. API Key Setup
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(page_title="AI PM Assessor", page_icon="💼")
st.title("💼 AI Business Value Assessor")
st.write("Upload a resume or report for a strict [Business Impact] evaluation.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Processing document logic..."):
        try:
            loader = PyPDFLoader("temp.pdf")
            raw_pages = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
            pages = text_splitter.split_documents(raw_pages)
            
            try:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_type="retrieval_document")
            except:
                try:
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")
                except:
                    embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001")

            vectorstore = FAISS.from_documents(documents=pages, embedding=embeddings)
            
       
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
            
            prompt_template = """
            You are a Senior AI Product Director. Evaluate the following document based on Business ROI and Storytelling.
            Context: {context}
            User Question: {question}
            Your Evaluation:
            """
            
            PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm, 
                retriever=vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": PROMPT} 
            )
            
            st.success("Analysis complete! The AI Director is ready.")
            
            user_question = st.text_input("Ask a business-focused question:")
            if user_question:
                response = qa_chain.invoke(user_question)
                st.write("### 💼 Evaluation Report:")
                st.write(response["result"])
                    
        except Exception as e:
            st.error("⚠️ AI engine encountered an interface mismatch.")
            st.info(f"Developer Error Log: {e}")
