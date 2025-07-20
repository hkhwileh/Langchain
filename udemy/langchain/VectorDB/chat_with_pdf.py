from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,OpenAI
from langchain_community.vectorstores import FAISS
import os
load_dotenv()

if __name__ =="__main__":
    pdf_path = "/Users/dc-hassan/Desktop/AI/My Repo/SampleCode/udemy/langchain/VectorDB/verdicts.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=30,separator="\n")
    doc = text_splitter.split_documents(documents)
    print(doc[2])

    #start embedding
    embeddings = OpenAIEmbeddings(openai_api_type=os.getenv("OPENAI_API_KEY"))
    vectorstore= FAISS.from_documents(doc,embeddings)
    vectorstore.save_local("faiss_inex_react")