import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA

from config import EMBED_MODEL, LLM_MODEL

# Where the FAISS index (vector database) is stored on disk
INDEX_DIR = "index"

# ------------------------------
# Setup logging so we get nice console messages
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def load_qa_pipeline():
    """
    Load the FAISS index from disk and connect it to an Ollama LLM
    through LangChain's RetrievalQA.

    Flow:
    1. Load embeddings (needed to deserialize the index properly).
    2. Load FAISS index (vector database with your document chunks).
    3. Create a retriever (grabs top-k relevant docs per question).
    4. Wrap it all in RetrievalQA, powered by Ollama.
    """
    logging.info("Loading FAISS index...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    logging.info("Creating retriever (top 3 results per query)...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    logging.info(f"Loading Ollama model: {LLM_MODEL}")
    llm = OllamaLLM(model=LLM_MODEL)

    logging.info("Assembling RetrievalQA pipeline...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain


if __name__ == "__main__":
    # Initialize the pipeline at startup
    qa = load_qa_pipeline()
    print("Q&A pipeline ready. Type your questions or 'exit' to quit.\n")

    # Simple CLI loop for testing
    while True:
        query = input("Question: ")
        if query.lower() in ["exit", "quit", "q"]:
            break

        try:
            # Run the query through the pipeline
            result = qa.invoke(query)

            # Print the answer from the LLM
            print("\nAnswer:")
            print(result["result"])

            # Show which documents were used to answer
            print("\nSources:")
            for doc in result["source_documents"]:
                print("-", doc.metadata.get("source", "Unknown"))

            print("\n" + "-" * 50 + "\n")

        except Exception as e:
            logging.error(f"Error during query: {e}", exc_info=True)
            print("An error occurred while processing your query.")
