import os
import json
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

CHART_KEYWORDS = [
    "chart",
    "graph",
    "plot",
    "bar chart",
    "line chart",
    "pie chart",
    "scatter",
    "scatter plot",
    "histogram",
    "visualize",
    "visualization"
]
chart_prompt = ChatPromptTemplate.from_template("""
You are a data extraction engine.

Return ONLY valid JSON.

Use these keys:

chart_type
title
x
y
labels
values
x_label
y_label

If no chart can be created return exactly:

{{"chart_type":"none"}}

Context:
{context}

Request:
{question}
""")
chart_chain = (chart_prompt | llm | StrOutputParser())

def is_chart_request(question):
    lowered=question.lower()
    return any(word in lowered for word in CHART_KEYWORDS)

def _to_number(values):
    if not isinstance(values, list) or not values:
        return None
    numbers = []
    for value in values:
        try:
            numbers.append(float(str(value).replace(",","").strip()))
        except(TypeError, ValueError):
            return None
    return numbers

def generate_chart_json(context, question):
    try:
        raw=chart_chain.invoke({"context": context, "question": question})
    except Exception as e:
        print("chart chain failed:", e)
        return None

    raw=raw.strip()

    if raw.startswith("```"):
        raw=raw[3:]
        if raw.lower().startswith("json"):
            raw=raw[4:]
    if raw.endswith("```"):
        raw=raw[:-3]
    
    raw=raw.strip()
    start,end=raw.find("{"),raw.rfind("}")

    if start==-1 or end==-1:
        return None

    try:
        spec=json.loads(raw[start:end+1])
    except Exception as e:
        print("json parse failed:", e)
        return None
    if not isinstance(spec, dict):
        return None
    chart_type=str(spec.get("chart_type","")).lower().strip()
    if chart_type not in ["bar","line","scatter","pie","histogram"]:
        return None

    if chart_type=="pie":
        labels=spec.get("labels")
        values=_to_number(spec.get("values"))
        if not isinstance(labels, list) or values is None or len(labels)!=len(values):
            return None
    elif chart_type=="histogram":
        if _to_number(spec.get("values")) is None:
            return None
    else:
        x=spec.get("x")
        y=_to_number(spec.get("y"))
        if not isinstance(x, list) or y is None or len(x)!=len(y):
            return None

    return spec
def build_plotly_chart(spec):
    import plotly.graph_objects as go
    chart_type=spec["chart_type"]
    try:
        if chart_type=="bar":
            fig= go.Figure(go.Bar(x=spec["x"], y=_to_number(spec["y"])))
        elif chart_type=="line":
            fig= go.Figure(go.Scatter(x=spec["x"], y=_to_number(spec["y"]), mode='lines+markers'))
        elif chart_type=="scatter":
            fig= go.Figure(go.Scatter(x=spec["x"], y=_to_number(spec["y"]), mode='markers'))
        elif chart_type=="pie":
            fig= go.Figure(go.Pie(labels=spec["labels"], values=_to_number(spec["values"])))
        elif chart_type=="histogram":
            fig= go.Figure(go.Histogram(x=_to_number(spec["values"])))
        else:
            return None
        fig.update_layout(
            title=spec.get("title", "Chart"),
            xaxis_title=spec.get("x_label") or None,
            yaxis_title=spec.get("y_label") or None
        )
        return fig
    except Exception as e:
        print("plotly chart build failed:", e)
        return None

def display_chart(spec,key):
    if not spec:
        return 
    fig=build_plotly_chart(spec)
    if fig is None:
        return
    try:
        st.plotly_chart(fig, use_container_width=True, key=key)
    except Exception as e:
        st.error(f"Failed to display chart: {e}")

def answer_not_found(answer):
    lowered=answer.lower().replace("`","'")
    return "couldn't find that information" in lowered 

st.set_page_config(page_title="Gemini RAG - Document Question Answering", page_icon="🤖", layout="wide")
st.title("Gemini RAG - Document Question Answering")
st.write("Ask questions about the documents in the `docs/` folder. The AI will answer based on the content of those documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for i,message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"]=="assistant":
            display_chart(message.get("chart_spec"),key=f"chart_{i}")

question = st.chat_input("Ask a question about the documents...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    docs = retriever.invoke(question)
    context = format_docs(docs)

    answer = rag_chain.invoke({"context": context, "question": question})

    chart_spec = None
    if is_chart_request(question):
        chart_spec = generate_chart_json(context, question)

    if is_chart_request(question) and chart_spec is not None and answer_not_found(answer):
        answer=f"here is the {chart_spec.get('chart_type','chart')} chart generated from the documents."

    with st.chat_message("assistant"):
        st.markdown(answer)
        display_chart(chart_spec,key=f"chart-live-{len(st.session_state.messages)}")
        if is_chart_request(question) and chart_spec is  None:
            st.info("a chart was requested but no usable numeric data could be extracted from the documents.")
        if not answer_not_found(answer):
            st.markdown("**Sources**")
            shown=[]
            for doc in docs:
                source=doc.metadata.get("source","Unknown")
                if source not in shown:
                    st.write(f"- {source}")
                    shown.append(source)

    st.session_state.messages.append({"role": "assistant", "content": answer, "chart_spec": chart_spec})