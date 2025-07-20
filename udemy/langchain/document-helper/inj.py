import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, PodSpec

# Ensure your environment variables are set
# os.environ["PINECONE_API_KEY"] = "YOUR_API_KEY"
# os.environ["PINECONE_ENVIRONMENT"] = "YOUR_ENVIRONMENT" # e.g., "gcp-starter"
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

def ingest_docs():
    # 1. Load Documents
    loader = PyPDFLoader("your_document.pdf") # Replace with your actual document path
    documents = loader.load()
    print(f"loaded {len(documents)} documents")

    # 2. Split Documents into Chunks
    # This is crucial for managing the size of data sent to Pinecone
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs_to_add = text_splitter.split_documents(documents)
    print(f"Going to add {len(docs_to_add)} chunks to Pinecone")

    # 3. Initialize Embeddings
    embeddings = OpenAIEmbeddings()

    # 4. Initialize Pinecone
    index_name = "your-pinecone-index-name" # Replace with your index name
    # If using serverless, specify cloud and region. For pod-based, PodSpec.
    # For serverless:
    # pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"), environment=os.environ.get("PINECONE_ENVIRONMENT"))
    # For pod-based:
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    if index_name not in pc.list_indexes():
        # It's good practice to create the index if it doesn't exist
        # For serverless, use cloud and region:
        # pc.create_index(index_name, dimension=1536, metric="cosine", cloud="aws", region="us-east-1")
        # For pod-based:
        pc.create_index(index_name, dimension=1536, metric="cosine", spec=PodSpec(environment=os.environ.get("PINECONE_ENVIRONMENT")))


    # 5. Add Documents to Pinecone in Batches
    # The PineconeVectorStore.from_documents method handles batching internally
    # when you have a large number of documents.
    # The issue you're facing might be due to very large individual chunks,
    # or the default batch size of LangChain's Pinecone integration being too large
    # for your specific document characteristics.

    # To explicitly control batching, you can iterate and add in chunks:
    # A good starting point for batch size is around 100-200 documents,
    # but it depends heavily on the size of your embeddings and metadata.
    batch_size = 100  # Adjust this value based on experimentation
    for i in range(0, len(docs_to_add), batch_size):
        batch = docs_to_add[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(docs_to_add) + batch_size - 1) // batch_size} (documents {i} to {min(i + batch_size, len(docs_to_add)) - 1})")
        PineconeVectorStore.from_documents(
            batch,
            embeddings,
            index_name=index_name
        )
    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_docs()