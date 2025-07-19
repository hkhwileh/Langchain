
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import CharacterTextSplitter
import os

load_dotenv()
if __name__ == "__main__":

    loader = TextLoader(
        "/Users/dc-hassan/Desktop/AI/My Repo/SampleCode/udemy/langchain/VectorDB/mediumblog1.txt",
        encoding="utf-8"
    )
    document =loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = splitter.split_documents(documents=document)

    # Embeddings
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        chunk_size=1000,
        api_key=os.getenv("OPENAI_API_KEY"),
        dimensions=1536
    )
    PineconeVectorStore.from_documents(chunks,embedding,index_name=os.getenv("PINECONE_INDEX_NAME"))
