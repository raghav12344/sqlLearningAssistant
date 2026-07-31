# SQL Learning Assistant 🤖📊

An intelligent, **Retrieval-Augmented Generation (RAG)** powered assistant designed to parse official SQL documentation and provide developers, students, and analysts with context-aware answers, query syntax guidance, and database concept explanations.

---

## 🌟 Key Features

- **📖 Documentation RAG Pipeline:** Ingests and indexes official SQL documentation into a vector database for hyper-relevant retrieval.
- **🎯 Grounded Answers:** Generates accurate responses with minimal LLM hallucinations by restricting context to official technical docs.
- **💡 Syntax & Query Assistance:** Helps users construct optimized `SELECT`, `JOIN`, aggregation, and DDL/DML queries with clear explanations.
- **⚡ Fast Vector Search:** Rapidly retrieves precise doc sections matching user queries using semantic similarity.

---

## 🛠️ Tech Stack

- **Language:** Python
- **AI & RAG Framework:** LangChain / LlamaIndex
- **Embeddings & Vector Store:** ChromaDB / FAISS
- **LLM Integration:** Google Gemini API / OpenAI
- **Data Processing:** PyPDF / BeautifulSoup / Unstructured (for doc ingestion)

---

## 🏗️ How It Works (Architecture)

1. **Ingestion:** Official SQL documentation files/web pages are split into manageable chunks.
2. **Embedding:** Text chunks are transformed into vector embeddings using embedding models.
3. **Storage:** Embeddings are stored in a local vector database for high-speed similarity search.
4. **Retrieval & Generation:** When a user asks a query, the system retrieves the most relevant SQL documentation chunks and feeds them into the LLM as context to generate an accurate answer.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- An API key for your LLM provider (e.g., Google Gemini API or OpenAI API)

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/raghav12344/sqlLearningAssistant.git](https://github.com/raghav12344/sqlLearningAssistant.git)
   cd sqlLearningAssistant

2. **Create and activate a virtual environment:**
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

4. **Environment Setup**
   Create a .env file in the root directory and add your API credentials:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key_here
   
5. **Run the Assistant:**
   ```bash
   Streamlit run app.py
