from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import ReadTheDocsLoader

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def ingest_doc():
    loader = ReadTheDocsLoader("/Users/dc-hassan/Desktop/api.python.langchain.com/en/latest")
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs","https:/")
        doc.metadata.update({"source":new_url})
    
    print(f"Going to add {len(documents)} to pinecoin")
    PineconeVectorStore.from_documents(documents,embeddings,index_name="langchain-doc-index")
    print("*****Loading to vectorstore done ******")


if __name__ == "__main__":
    ingest_doc()
