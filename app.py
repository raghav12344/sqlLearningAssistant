import os
import fitz
from dotenv import load_dotenv
import torch
import streamlit as st

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

# 1. App Configuration
st.set_page_config(
    page_title="Gemini RAG",
    page_icon="🤖",
    layout="wide"
)

st.title("Gemini RAG - Document Question Answering")

# Load Environment Variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is not set in the environment variables.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Constants
DATA_FOLDER = "docs"
DB_FOLDER = "chromadb"
LLM_NAME = "gemini-3.5-flash-lite"  # Updated to official standard model name
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
TOP_K = 8

os.makedirs(DATA_FOLDER, exist_ok=True)

# 2. Cached Resource Initialization
@st.cache_resource
def get_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )

@st.cache_resource
def load_or_create_vectorstore(_embeddings):
    # Check if DB exists and is non-empty
    if os.path.exists(DB_FOLDER) and len(os.listdir(DB_FOLDER)) > 0:
        return Chroma(
            persist_directory=DB_FOLDER,
            embedding_function=_embeddings
        )

    # Process documents ONLY if DB does not exist
    documents = []
    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]

    if not pdf_files:
        st.warning(f"No PDF files found in '{DATA_FOLDER}' directory. Please add PDFs and restart.")
        return None

    for file in pdf_files:
        pdf_path = os.path.join(DATA_FOLDER, file)
        pdf = fitz.open(pdf_path)

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
    chunks = splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=_embeddings,
        persist_directory=DB_FOLDER
    )
    return vectordb

# Initialize Vectorstore
embeddings = get_embeddings()
vectordb = load_or_create_vectorstore(embeddings)

if vectordb is None:
    st.info("Add `.pdf` files to the `docs/` folder and refresh the page.")
    st.stop()

retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})
llm = ChatGoogleGenerativeAI(model=LLM_NAME)

prompt = ChatPromptTemplate.from_template("""
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
        f"[Source: {doc.metadata['source']} | Page: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
        for doc in docs
    )

rag_chain = prompt | llm | StrOutputParser()

# 3. Chat Interface
st.write("Ask questions based on the uploaded documents. The AI will answer only from the provided context.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Query Processing
question = st.chat_input("Ask a question about the documents...")

if question:
    # Render user prompt
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Retrieval and Generation
    with st.spinner("Searching documents..."):
        docs = retriever.invoke(question)
        context = format_docs(docs)

        answer = rag_chain.invoke(
            {
                "context": context,
                "question": question
            }
        )

    # Render assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)

        # Show source citations if information was found
        if "couldn't find that information" not in answer.lower():
            st.markdown("**Sources:**")
            shown = set()
            for doc in docs:
                source = doc.metadata.get("source")
                page = doc.metadata.get("page")
                source_str = f"{source} (Page {page})"
                if source_str not in shown:
                    st.write(f"- {source_str}")
                    shown.add(source_str)

    st.session_state.messages.append({"role": "assistant", "content": answer})