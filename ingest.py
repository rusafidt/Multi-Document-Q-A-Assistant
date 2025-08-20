import os
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from config import EMBED_MODEL

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

DATA_DIR = "data"
INDEX_DIR = "index"


def load_documents():
    """
    Loads documents from the data directory.
    Supports PDF, TXT, and DOCX formats.
    Returns a list of LangChain Document objects.
    """
    docs = []
    if not os.path.exists(DATA_DIR):
        logging.warning("Data directory not found. Creating empty 'data/' folder.")
        os.makedirs(DATA_DIR)
        return docs

    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".txt"):
            loader = TextLoader(path)
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            logging.info(f"Skipping unsupported file: {file}")
            continue

        logging.info(f"Loading file: {file}")
        docs.extend(loader.load())

    logging.info(f"Total documents loaded: {len(docs)}")
    return docs


def build_index():
    """
    Builds or updates a FAISS vector index using Ollama embeddings.
    - If index exists, adds new documents incrementally.
    - Otherwise, creates a fresh index.
    """
    docs = load_documents()
    if not docs:
        raise ValueError("No documents found in 'data/' folder.")

    # Split docs into smaller chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    logging.info(f"Documents split into {len(chunks)} chunks.")

    # Initialize Ollama embeddings
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    if os.path.exists(INDEX_DIR):
        logging.info("Existing index found. Updating with new documents...")
        vectorstore = FAISS.load_local(
            INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(chunks)
    else:
        logging.info("No index found. Creating new index...")
        vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(INDEX_DIR)
    logging.info(f"Index saved to: {INDEX_DIR}")

    return f"Index updated & saved to '{INDEX_DIR}'"


if __name__ == "__main__":
    import sys
    try:
        msg = build_index()
        print(msg)
    except Exception as e:
        logging.error(f"Failed to build index: {e}", exc_info=True)
        sys.exit(1)
