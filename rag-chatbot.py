import os
from dotenv import load_dotenv
import streamlit as st
import pdfplumber as pp
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
# creating the user inferface
st.header("My chatbot")
with st.sidebar:
    st.title("Upload your file")
    file = st.file_uploader("Upload only pdf here and start asking questions.", type="pdf")

# reading/extracting the pdf file
if file is not None:
    with pp.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text+=page.extract_text() + "\n"
    
    # st.write(text)

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)
    # st.write(chunks)    

    # generating embeddings with Gemini
    # generating embeddings
    # embeddings = GoogleGenerativeAIEmbeddings(
    #     model="models/text-embedding-004" 
    # )
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004" 
    )

    # Store embeddings in vector db
    vector_store = FAISS.from_texts(chunks, embeddings)

    # --- STREAMLIT VERIFICATION ---
    st.success(f"Success! I have generated {vector_store.index.ntotal} embeddings.")

    # --- STREAMLIT VERIFICATION ---
    test_results = vector_store.similarity_search("What is this document about?", k=1)

    st.write("### 🔍 Test Search Result:")
    st.info(test_results[0])

