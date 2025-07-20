from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import os
load_dotenv()

from langchain import hub

if __name__ =="__main__":
    pdf_path = "/Users/dc-hassan/Desktop/AI/My Repo/SampleCode/udemy/langchain/VectorDB/verdicts.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=30,separator="\n")
    doc = text_splitter.split_documents(documents)
    print(doc[2])

    #start embedding
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore= FAISS.from_documents(doc,embeddings)
    vectorstore.save_local("faiss_inex_react")
    new_vectorstore = FAISS.load_local("faiss_inex_react",embeddings,allow_dangerous_deserialization=True)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chain = create_stuff_documents_chain(OpenAI(),retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(new_vectorstore.as_retriever(),combine_docs_chain=combine_docs_chain)
    res = retrival_chain.invoke(input={"input":"ما هو رقم الطعن؟"})
    print(res["answer"])
