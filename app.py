import os
import fitz

import streamlit as st

st.set_page_config(
    page_title="Gemini RAG",
    page_icon="🤖",
    layout="wide"
)

st.title("Gemini RAG - Document Question Answering")
st.write("Loading application...")

from dotenv import load_dotenv

from docling.document_converter import DocumentConverter

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma

from langchain_google_genai import(
    ChatGoogleGenerativeAI
)

from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY is None:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY

DATA_FOLDER="docs"
DB_FOLDER="chromadb"

LLM_NAME="gemini-3.5-flash-lite"

CHUNK_SIZE=4000
CHUNK_OVERLAP=500
TOP_K=8

os.makedirs(DATA_FOLDER,exist_ok=True)

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("\n")
print(f"Using device: {device}")
print("\n")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)

if os.path.exists(DB_FOLDER) and len(os.listdir(DB_FOLDER))>0:
  print("Loading existing ChromaDB...")

  vectordb=Chroma(
      persist_directory=DB_FOLDER,
      embedding_function=embeddings
  )
else:
  print("No existing database found.")
  print("Building ChromaDB...")



documents = []

for file in os.listdir(DATA_FOLDER):

    if not file.endswith(".pdf"):
        continue

    pdf = fitz.open(os.path.join(DATA_FOLDER, file))

    for page_num, page in enumerate(pdf, start=1):
        text = page.get_text()

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file,
                        "page": page_num
                    }
                )
            )

    pdf.close()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
chunks=splitter.split_documents(documents)
print(f"\n total chunks: {len(chunks)}")

vectordb=Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_FOLDER
)
print("\n ChromaDB created successfully")

retriever=vectordb.as_retriever(
    search_kwargs={"k":TOP_K}
)

llm=ChatGoogleGenerativeAI(
    model=LLM_NAME,
)


prompt=ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Answer ONLY from the given context.

If the answer is not available in the context, say:
"I couldn't find that information in the provided documents."

Keep the answer concise and accurate.

Context:
{context}

Question:
{question}

Answer:
""")

def format_docs(docs):
  return "\n\n".join(
      f"[Source: {doc.metadata['source']}]\n{doc.page_content}"
      for doc in docs
  )

rag_chain=(
    prompt
    | llm
    | StrOutputParser()
)


st.write("Ask questions based on the uploaded documents. The AI will answer only from the provided context.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about the documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    docs = retriever.invoke(question)
    context = format_docs(docs)

    answer = rag_chain.invoke(
        {
            "context": context,
            "question": question
        }
    )



    with st.chat_message("assistant"):
        st.markdown(answer)

    if "I couldn't find that information in the provided documents." not in answer.lower():
        st.markdown("**Sources:**")
        shown=[]
        for doc in docs:
            source = doc.metadata["source"]
            if source not in shown:
                st.write(f"- {source}")
                shown.append(source)

    st.session_state.messages.append({"role": "assistant", "content": answer})