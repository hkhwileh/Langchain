import os
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
# LangChain & OpenAI
dotenv.load_dotenv()

if __name__ == "__main__":
    print("🚀 Hello LangChain VectorDB Retrieval!")

    # Pinecone LangChain wrapper

    # Pinecone SDK v3+
    from pinecone import Pinecone as PineconeClient

    # Init Pinecone client and get index object
    pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "test"

    # ✅ Get actual Pinecone index object (not just the string)
    index = pc.Index(index_name)

    print("✅ Successfully connected to Pinecone Index.")
    embedding = OpenAIEmbeddings()
    llm = ChatOpenAI()

    chain = PromptTemplate.from_template(template="What is Pinecone in machine learning?")|llm
    query = "What is Pinecone in machine learning?"
    res = chain.invoke({"query": "What is Pinecone in machine learning?"})
    print(f"Response: {res.content}")

    vectorstore = PineconeVectorStore(index=index, embedding=embedding, text_key="text")

    retrival_qa_chain = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm,retrival_qa_chain)
    retrival_chain = create_retrieval_chain(
        vectorstore=vectorstore.as_retriever(),
        combine_docs_chain=combine_docs_chain
    )

    res_retrieval = retrival_chain.invoke(input={"input":query})
    print(f"Response: {res_retrieval['output']}")