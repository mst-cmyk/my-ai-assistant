import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter # Added: Text Splitter for handling large limits

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
            # 1. Load PDF
            loader = PyPDFLoader("temp.pdf")
            raw_pages = loader.load()
            
            # Check if PDF is empty or purely scanned images
            if not raw_pages or len(raw_pages[0].page_content.strip()) == 0:
                st.warning("⚠️ Oops! It seems this PDF is a scanned image or empty. Please upload a text-based PDF.")
            else:
                # 2. Split text into safe chunks to avoid API limits
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                pages = text_splitter.split_documents(raw_pages)
                
                # 3. Create Vector Store
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vectorstore = FAISS.from_documents(documents=pages, embedding=embeddings)
                
                # 4. Setup LLM and Prompt
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
                
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
                
                user_question = st.text_input("Ask me a question (e.g., Evaluate the business potential of this candidate):")
                if user_question:
                    response = qa_chain.invoke(user_question)
                    st.write("### 💼 Evaluation Report:")
                    st.write(response["result"])
                    
        except Exception as e:
            # PM-level Error Handling
            st.error("⚠️ The AI engine is currently experiencing high traffic or the document is too complex. Please try again later or upload a shorter document.")
            st.info(f"Developer Error Log: {e}")
