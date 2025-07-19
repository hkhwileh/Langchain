"""
this code will doe the following:
1-embedding user query
2-semantic search
3-prompt augmentation
4-generation
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough

def formate_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__=="__main__":
    print("Retringin..")
    embedding = OpenAIEmbeddings()
    llm = ChatOpenAI()

    query = "what is pinecone in machine learning?"
    chain = PromptTemplate.from_template(template=query) | llm
    #result = chain.invoke(input={})
    #print(result.content)

    vectorstore = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX_NAME"),embedding=embedding)

    retrival_qa_caht_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_doc_chain = create_stuff_documents_chain(llm,retrival_qa_caht_prompt )
    retrival_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(),combine_docs_chain=combine_doc_chain)

    template = """Use the following pieces of context to answer """

    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer concise as possible.
    always say "thanks for asking!" at the end of the answer.

    {context}

    Question: {question}
    Helpful answer:
    """
    custom_rag_prompt = PromptTemplate.format_prompt(template)
    rag_chain = (
        {"context":vectorstore.as_retriever() | formate_docs, "question":RunnablePassthrough()}
        | custom_rag_prompt
        |llm
    )

    res = rag_chain.invode(query)
    print(res)

