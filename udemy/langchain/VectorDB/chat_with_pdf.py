from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

if __name__ =="__main__":
    pdf_path = "/Users/dc-hassan/Desktop/AI/My Repo/SampleCode/udemy/langchain/VectorDB/verdicts.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=30,separator="\n")
    doc = text_splitter.split_documents(documents)
    print(doc[2])