# 📚 Multi-Document Q&A Assistant

A local-first Question Answering assistant powered by [Ollama](https://ollama.com).  
This project lets you **ingest multiple documents** (PDF, TXT, DOCX), build a FAISS index with embeddings, and query them through a **FastAPI backend**.

---

## 🚀 Features
- Ingest and index documents automatically (no manual step needed)
- Local embeddings and LLMs via **Ollama**
- Query through a **FastAPI** endpoint (`/query`)
- Returns both **answers** and **document snippets** for context
- Incremental ingestion (new documents can be added later)

---

## ⚙️ Setup

### 1. Install Ollama
Follow instructions from [Ollama Install](https://ollama.com/download) and start the Ollama service:
```bash
ollama serve
ollama pull mxbai-embed-large
ollama pull llama3.2
```

### 2. Create Conda environment
```bash
conda env create -f environment.yml
conda activate multi_doc_qa
```

### 3. Add document
Place your .pdf, .txt, or .docx files into the data/ folder.

### 4. Run the server
```bash
uvicorn main:app --reload
```

## 💡 Usage

### Query via Postman or cURL

**POST request to:** 

[http://127.0.0.1:8000/query](http://127.0.0.1:8000/query)

###Body (JSON):

**Request**
```json
{
  "question": "What is this document about?"
}
```

**Response**
```json
{
  "answer": "The document discusses privacy in chatbots.",
  "sources": [
    {
      "source": "data/paper1.pdf",
      "snippet": "Despite the increasing prevalence of AI-powered therapy chatbots..."
    }
  ]
}
```

## 📝 Notes
--> Ensure ollama serve is running before starting the server.

--> Add your documents into the data/ folder before running python main.py.

--> The data/ and index/ folders are excluded from Git via .gitignore.
