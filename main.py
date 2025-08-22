import os
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA

from config import EMBED_MODEL, LLM_MODEL
from models.pydantic_models import QueryRequest, QueryResponse
from ingest import build_index   # reuse ingestion script to build index

# ------------------------------
# Setup logging so we can see what’s happening in the console
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

INDEX_DIR = "index"  # folder where FAISS index will be saved

# ------------------------------
# Create the FastAPI app
# ------------------------------
app = FastAPI(title="Multi-Document Q&A Assistant")


def load_qa_pipeline():
    """
    Build or load the FAISS index, then create a RetrievalQA pipeline.
    If the index doesn’t match our embeddings (dimension mismatch), 
    we’ll wipe it and rebuild from scratch.
    """
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # If no index exists yet → build one before continuing
    if not os.path.exists(INDEX_DIR) or not os.listdir(INDEX_DIR):
        logging.info("No index found. Running ingestion to build index...")
        build_index()

    try:
        logging.info("Loading FAISS index from disk...")
        vectorstore = FAISS.load_local(
            INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except AssertionError as e:
        logging.warning("Index dimension mismatch detected. Rebuilding index...")
        # If embeddings don’t match → delete and rebuild index
        import shutil
        shutil.rmtree(INDEX_DIR)
        build_index()
        vectorstore = FAISS.load_local(
            INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # Turn the vectorstore into a retriever that grabs top-3 matches
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    logging.info(f"Loading Ollama model: {LLM_MODEL}")
    llm = OllamaLLM(model=LLM_MODEL)

    # Create a RetrievalQA pipeline:
    #   1. Retriever finds the most relevant docs
    #   2. LLM generates an answer based on them
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain


# Load the Q&A pipeline as soon as the app starts
qa_chain = load_qa_pipeline()


@app.post("/query", response_model=QueryResponse)
def query_docs(request: QueryRequest):
    """
    Endpoint: Accept a question, run it through our RetrievalQA chain,
    and return both the generated answer + the document snippets used.
    """
    try:
        logging.info(f"Received query: {request.question}")
        result = qa_chain.invoke(request.question)

        # Extract the model’s answer
        answer = result["result"]

        # Collect sources with a short preview snippet
        sources = []
        for doc in result["source_documents"]:
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "snippet": doc.page_content[:200]  # first 200 chars
            })

        return QueryResponse(answer=answer, sources=sources)

    except Exception as e:
        logging.error(f"Error while processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logging.info("Starting FastAPI server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
