# Beginner Guide: `rag-chatbot.py`

This document explains how `rag-chatbot.py` works in beginner-friendly language.

## What This App Does

`rag-chatbot.py` is a simple PDF question-answering chatbot.

The user uploads a PDF file, types a question, and the app answers using information from that PDF. It does this with a common AI pattern called **RAG**, which stands for **Retrieval-Augmented Generation**.

In plain English:

1. Read the PDF.
2. Break the PDF text into smaller pieces.
3. Convert those pieces into searchable numeric representations called embeddings.
4. Store those embeddings in a vector database.
5. When the user asks a question, search for the most relevant PDF pieces.
6. Send those pieces plus the question to an LLM.
7. Show the answer in the Streamlit app.

## Main Libraries Used

The script imports several libraries:

```python
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
```

Here is what they are used for:

- `dotenv`: loads secret values like API keys from a `.env` file.
- `streamlit`: creates the web app interface.
- `pdfplumber`: reads and extracts text from PDF files.
- `RecursiveCharacterTextSplitter`: splits long text into smaller chunks.
- `GoogleGenerativeAIEmbeddings`: creates embeddings using Google's Gemini embedding model.
- `FAISS`: stores and searches embeddings locally.
- `ChatGoogleGenerativeAI`: connects to the Gemini chat model.
- `ChatPromptTemplate`: creates the prompt sent to the AI model.
- `StrOutputParser`: converts the model response into plain text.
- `RunnablePassthrough`: passes the user's question through the LangChain chain.

## Step-by-Step Code Explanation

### 1. Load Environment Variables

```python
load_dotenv()
```

This loads values from the `.env` file into the program environment.

For this app, the `.env` file likely contains a Google API key. The Google AI libraries need that key to call Gemini models.

### 2. Create the Streamlit Interface

```python
st.header("My chatbot")
with st.sidebar:
    st.title("Upload your file")
    file = st.file_uploader("Upload only pdf here and start asking questions.", type="pdf")
```

This creates a simple web page:

- The main page has the heading `My chatbot`.
- The sidebar lets the user upload a PDF file.
- The uploader only accepts files with the PDF type.

At this point, the app is waiting for the user to upload a document.

### 3. Run the Main Logic Only After a PDF Is Uploaded

```python
if file is not None:
```

This line checks whether the user uploaded a file.

If no file is uploaded, the chatbot does not process anything. If a PDF is uploaded, the rest of the app runs.

### 4. Extract Text from the PDF

```python
with pp.open(file) as pdf:
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"
```

This opens the uploaded PDF and loops through each page.

For every page:

- `page.extract_text()` extracts the text from that page.
- The extracted text is added to the `text` variable.
- A newline is added after each page's text.

After this step, `text` contains the full text of the uploaded PDF.

### 5. Split the PDF Text into Chunks

```python
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_text(text)
```

Large documents are difficult to send directly to an AI model. So the app breaks the PDF text into smaller chunks.

Important settings:

- `chunk_size=1000`: each chunk should be around 1000 characters.
- `chunk_overlap=200`: each chunk shares about 200 characters with the next chunk.

The overlap helps prevent important context from being lost between chunks.

The `separators` list tells LangChain where it should try to split the text first. It prefers larger natural breaks like paragraphs and lines before splitting by spaces or individual characters.

### 6. Create Embeddings

```python
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)
```

An embedding is a numeric representation of text.

Texts with similar meaning usually have embeddings that are close to each other. This makes it possible to search for PDF chunks by meaning instead of only by exact words.

This app uses Google's `models/gemini-embedding-001` model to create embeddings.

### 7. Store Chunks in FAISS

```python
vector_store = FAISS.from_texts(chunks, embeddings)
```

This creates a FAISS vector store from the text chunks.

FAISS is used as a local vector database. It stores the embeddings and can quickly find chunks that are semantically similar to a user's question.

### 8. Ask the User for a Question

```python
user_question = st.text_input("Type your question here!")
```

This adds a text box where the user can type a question about the uploaded PDF.

### 9. Format Retrieved Documents

```python
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])
```

The retriever returns document objects. The LLM needs plain text.

This helper function takes the retrieved documents and combines their text into one string.

### 10. Create the Retriever

```python
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4}
)
```

The retriever searches the vector store for chunks related to the user's question.

Important settings:

- `search_type="mmr"` uses Maximal Marginal Relevance.
- `k=4` means it retrieves 4 chunks.

MMR tries to return results that are both relevant and diverse, so the LLM receives useful context without too much repeated information.

### 11. Define the Gemini Chat Model

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.3,
    max_tokens=1000,
)
```

This creates the LLM that will answer the user's question.

Settings:

- `model="gemini-3-flash-preview"`: the Gemini chat model used for answering.
- `temperature=0.3`: keeps answers more focused and less random.
- `max_tokens=1000`: limits how long the response can be.

### 12. Create the Prompt

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "..."),
    ("human", "{question}")
])
```

The prompt tells the AI how to behave.

The system message says:

- Answer questions about a PDF document.
- Use only the provided context.
- Include useful details.
- Summarize long information when needed.
- If the answer is not in the context, say so politely.

The prompt includes this placeholder:

```text
{context}
```

That is where the retrieved PDF chunks are inserted.

It also includes:

```text
{question}
```

That is where the user's question is inserted.

### 13. Build the LangChain Chain

```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

This is the main RAG pipeline.

Here is what happens inside the chain:

1. The user's question is sent to the retriever.
2. The retriever finds relevant PDF chunks.
3. `format_docs` turns those chunks into plain text.
4. The original question is passed through unchanged.
5. The context and question are inserted into the prompt.
6. The prompt is sent to Gemini.
7. The response is converted into a normal string.

### 14. Generate and Display the Answer

```python
if user_question:
    llm_response = chain.invoke(user_question)

    st.write(llm_response)
```

This runs only when the user types a question.

The app:

1. Sends the question into the RAG chain.
2. Gets the answer from Gemini.
3. Displays the answer using Streamlit.

## Full Flow Summary

The app works like this:

```text
Upload PDF
   |
Extract text with pdfplumber
   |
Split text into chunks
   |
Create embeddings for chunks
   |
Store embeddings in FAISS
   |
User asks a question
   |
Retrieve relevant chunks
   |
Send chunks + question to Gemini
   |
Display answer in Streamlit
```

## What RAG Means in This Code

RAG means **Retrieval-Augmented Generation**.

In this app:

- **Retrieval** happens when FAISS finds relevant PDF chunks.
- **Augmented** means those chunks are added to the prompt as context.
- **Generation** happens when Gemini writes the final answer.

This approach helps the chatbot answer based on the uploaded PDF instead of relying only on the model's general knowledge.

## Important Variables

| Variable | Meaning |
| --- | --- |
| `file` | The PDF uploaded by the user |
| `text` | All extracted text from the PDF |
| `text_splitter` | Tool that splits long text into chunks |
| `chunks` | Smaller pieces of the PDF text |
| `embeddings` | Model used to convert text into vectors |
| `vector_store` | FAISS database storing the chunk embeddings |
| `user_question` | The question typed by the user |
| `retriever` | Searches for relevant chunks |
| `llm` | Gemini chat model used to generate answers |
| `prompt` | Instructions and context sent to the LLM |
| `chain` | The full RAG pipeline |
| `llm_response` | The final answer displayed to the user |

## Notes for Beginners

- The app rebuilds the vector store every time a PDF is uploaded and the script reruns.
- The chatbot can only answer well if the PDF text is extracted correctly.
- Scanned image PDFs may not work because `pdfplumber` extracts text, not text from images.
- The prompt tells the model to use only the retrieved PDF context.
- The quality of the answer depends on the PDF text, chunking, retrieval, and the LLM response.

## Possible Improvements

Here are some beginner-friendly improvements you could add later:

- Show a message after the PDF is processed successfully.
- Handle pages where `page.extract_text()` returns `None`.
- Cache the vector store so it is not rebuilt on every Streamlit rerun.
- Show the source chunks used to answer the question.
- Add error handling for missing API keys.
- Allow multiple PDFs.
- Add a loading spinner while the app processes the PDF or generates an answer.

