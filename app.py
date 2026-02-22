import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Securely load API Key
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(page_title="AI PM Resume Screener", page_icon="💼")
st.title("💼 AI Business Value Assessor")
st.write("Upload a resume or report. I will evaluate it strictly based on [Business Impact] and [Product Sense].")

uploaded_file = st.file_uploader("Please upload a PDF file", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Analyzing from a business perspective (Powered by FAISS)..."):
        try:
            loader = PyPDFLoader("temp.pdf")
            raw_pages = loader.load()
            
            if not raw_pages or len(raw_pages[0].page_content.strip()) == 0:
                st.warning("⚠️ Oops! It seems this PDF is a scanned image or empty. Please upload a text-based PDF.")
            else:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
                pages = text_splitter.split_documents(raw_pages)
                
                # 👉 修正 1：移除所有 model 前缀，让 SDK 自动映射
                embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001")
                vectorstore = FAISS.from_documents(documents=pages, embedding=embeddings)
                
                # 👉 修正 2：使用最基础的定义，不强制指定 version，避开 404 坑
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
                
                prompt_template = """
                You are an extremely strict Senior AI Product Director. Evaluate the following document:
                1. [Business Value]: Impact on ROI or efficiency.
                2. [0 to 1 Execution]: Evidence of leading products.
                3. [Storytelling]: Ability to explain complex data as business value.
                
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
                
                st.success("Analysis complete! I have put on my AI Product Director glasses.")
                
                user_question = st.text_input("Ask me a question:")
                if user_question:
                    response = qa_chain.invoke(user_question)
                    st.write("### 💼 Evaluation Report:")
                    st.write(response["result"])
                    
        except Exception as e:
            st.error("⚠️ The AI engine is struggling with Google's API connection.")
            st.info(f"Developer Error Log: {e}")
