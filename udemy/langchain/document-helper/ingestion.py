from dotenv import load_dotenv
load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import time

def ingest_docs():
    loader = ReadTheDocsLoader("/Users/dc-hassan/Desktop/api.python.langchain.com/en/latest")
    
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    
    # Reduce chunk size to prevent size limit issues
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Reduced from 1000
        chunk_overlap=50,
        length_function=len,
    )
    documents = text_splitter.split_documents(raw_documents)
    
    # Fix the URL replacement (you had a typo "Desltop" instead of "Desktop")
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("/Users/dc-hassan/Desktop", "https:/")
        doc.metadata.update({"source": new_url})
    
    print(f"Going to add {len(documents)} to Pinecone")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Process documents in smaller batches to avoid size limits
    batch_size = 50  # Process 50 documents at a time
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1} of {(len(documents) + batch_size - 1)//batch_size}")
        
        try:
            if i == 0:
                # Create the vector store with the first batch
                vector_store = PineconeVectorStore.from_documents(
                    batch, embeddings, index_name="langchain-doc-index"
                )
            else:
                # Add subsequent batches to existing vector store
                vector_store.add_documents(batch)
            
            print(f"Successfully added batch {i//batch_size + 1}")
            # Add a small delay between batches to avoid rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            # Continue with next batch instead of stopping
            continue
    
    print("****Loading to vectorstore done ***")

if __name__ == "__main__":
    ingest_docs()