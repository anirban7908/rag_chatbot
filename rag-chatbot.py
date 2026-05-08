import os
from dotenv import load_dotenv
import streamlit as st
import pdfplumber as pp
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


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
        model="models/gemini-embedding-001" 
    )

    # Store embeddings in vector db
    vector_store = FAISS.from_texts(chunks, embeddings)

    # --- STREAMLIT VERIFICATION ---
    # st.success(f"Success! I have generated {vector_store.index.ntotal} embeddings.")

    # # --- STREAMLIT VERIFICATION ---
    # test_results = vector_store.similarity_search("What is this document about?", k=1)

    # st.write("### 🔍 Test Search Result:")
    # st.info(test_results[0])

    # Get user input
    user_question = st.text_input("Type your question here!")

    #generate answer
    #question -> embeddings -> similiairty search -> results to LLM -> response (CHAIN)
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k":4}
    )

    #define the LLM and prompts
    # define the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview", 
        temperature=0.3,
        max_tokens=1000,
    )

    # Provide the prompts (Optimized for Gemini)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "You are a helpful assistant answering questions about a PDF document.\n\n"
        "Guidelines:\n"
        "1. Provide complete, well-explained answers using the context below.\n"
        "2. Include relevant details, numbers, and explanations to give a thorough response.\n"
        "3. If the context mentions related information, include it to give a fuller picture.\n"
        "4. Only use information from the provided context - do not use outside knowledge.\n"
        "5. Summarize long information, ideally in bullets where needed.\n"
        "6. If the information is not in the context, say so politely.\n\n"
        "<context>\n{context}\n</context>"),
        ("human", "{question}")
    ])
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        |StrOutputParser()
    )

    if user_question:
        llm_response = chain.invoke(user_question)

        st.write(llm_response)
