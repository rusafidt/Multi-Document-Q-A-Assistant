import os
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from config import EMBED_MODEL

# ------------------------------
# Setup logging so we can see what's happening
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Where raw files live and where the FAISS index will be stored
DATA_DIR = "data"
INDEX_DIR = "index"


def load_documents():
    """
    Look inside the `data/` folder and load all supported documents.
    Supported types: PDF, TXT, DOCX.
    Returns a list of LangChain Document objects ready for embedding.
    """
    docs = []

    # If no data folder exists yet, make one (but it will be empty)
    if not os.path.exists(DATA_DIR):
        logging.warning("Data directory not found. Creating empty 'data/' folder.")
        os.makedirs(DATA_DIR)
        return docs

    # Loop through all files and load them using the right loader
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
    Build or update the FAISS vector index using Ollama embeddings.

    - If an index already exists → we add new documents into it.
    - If no index exists yet → we create one from scratch.
    """
    docs = load_documents()
    if not docs:
        raise ValueError("No documents found in 'data/' folder.")

    # Break documents into smaller overlapping chunks
    # (helps retrieval because long docs are harder to search directly)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    logging.info(f"Documents split into {len(chunks)} chunks.")

    # Set up Ollama embeddings (turns text into vectors)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    if os.path.exists(INDEX_DIR):
        # If index already exists, update it with any new content
        logging.info("Existing index found. Updating with new documents...")
        vectorstore = FAISS.load_local(
            INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(chunks)
    else:
        # If no index exists, create one from scratch
        logging.info("No index found. Creating new index...")
        vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save the index to disk so we can reuse it later
    vectorstore.save_local(INDEX_DIR)
    logging.info(f"Index saved to: {INDEX_DIR}")

    return f"Index updated & saved to '{INDEX_DIR}'"


# Run directly: build index from whatever is in `data/`
if __name__ == "__main__":
    import sys
    try:
        msg = build_index()
        print(msg)
    except Exception as e:
        logging.error(f"Failed to build index: {e}", exc_info=True)
        sys.exit(1)
