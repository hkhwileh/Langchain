from dotenv import load_dotenv

load_dotenv()

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI , OpenAIEmbeddings

INDEX_NAME = "langchain-doc-index"

def run_llm(query:str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME,embedding=embeddings) 
    chat = ChatOpenAI(verbose=True,temperature=0)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chain = create_stuff_documents_chain(chat,retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(docsearch.as_retriever(),combine_docs_chain=combine_docs_chain)
    res = retrival_chain.invoke(input={"input":query})
    print(res["answer"])

if __name__=="__main__":
    run_llm("what is langchain components")
