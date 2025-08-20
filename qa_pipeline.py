import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA

from config import EMBED_MODEL, LLM_MODEL

INDEX_DIR = "index"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def load_qa_pipeline():
    """
    Load the FAISS index and prepare a RetrievalQA pipeline.
    - Loads embeddings (for deserialization).
    - Wraps Ollama LLM for question answering.
    """
    logging.info("Loading FAISS index...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    logging.info("Creating retriever...")
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
    qa = load_qa_pipeline()
    print("Q&A pipeline ready. Type your questions or 'exit' to quit.\n")

    while True:
        query = input("Question: ")
        if query.lower() in ["exit", "quit", "q"]:
            break

        try:
            result = qa.invoke(query)
            print("\nAnswer:")
            print(result["result"])

            print("\nSources:")
            for doc in result["source_documents"]:
                print("-", doc.metadata.get("source", "Unknown"))

            print("\n" + "-" * 50 + "\n")

        except Exception as e:
            logging.error(f"Error during query: {e}", exc_info=True)
            print("An error occurred while processing your query.")
