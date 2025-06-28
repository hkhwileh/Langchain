import os
import dotenv

# LangChain & OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# Pinecone LangChain wrapper
from langchain_pinecone import PineconeVectorStore

# Pinecone SDK v3+
from pinecone import Pinecone as PineconeClient

# Load .env
dotenv.load_dotenv()

# Init Pinecone client and get index object
pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "test"

# ✅ Get actual Pinecone index object (not just the string)
index = pc.Index(index_name)

if __name__ == "__main__":
    print("🚀 Hello LangChain VectorDB Ingestion!")

    # Load and split documents
    loader = TextLoader(
        "/Users/dc-hassan/Desktop/AI/My Repo/SampleCode/udemy/langchain/VectorDB/mediumblog1.txt",
        encoding="utf-8"
    )
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents.")

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(documents)
    print(f"✅ Split into {len(texts)} chunks.")

    # Embeddings
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        chunk_size=1,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # ✅ Use the real Pinecone Index object
    vector_store = PineconeVectorStore(
        index=index,  # ✅ index object, not a string
        embedding=embedding,
        text_key="text"
    )

    vector_store.add_documents(texts)

    print("✅ Successfully ingested documents into Pinecone via LangChain.")
