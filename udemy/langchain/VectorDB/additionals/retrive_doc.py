from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from typing import List


# Custom retriever for static document testing
class CustomRetriever(BaseRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        return [
            Document(
                page_content="Japan has a population of 126 million people.",
                metadata={"source": "https://en.wikipedia.org/wiki/Japan"},
            ),
            Document(
                page_content="Japanese people are very polite.",
                metadata={"source": "https://en.wikipedia.org/wiki/Japanese_people"},
            ),
            Document(
                page_content="United States has a population of 328 million people.",
                metadata={"source": "https://en.wikipedia.org/wiki/United_States"},
            ),
        ]


# QA Prompt
qa_prompt = PromptTemplate.from_template("""
Use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If the question is to request links, please only return the source links with no answer.
2. If you don't know the answer, don't try to make up an answer. Just say **I can't find the final answer but you may want to check the following links** and add the source links as a list.
3. If you find the answer, write the answer in a concise way and add the list of sources that are **directly** used to derive the answer. Exclude the sources that are irrelevant to the final answer.

{context}

Question: {question}
Helpful Answer:
""")


# Main pipeline
def main():
    retriever = CustomRetriever()

    llm = ChatOpenAI(temperature=0)

    # Create a combine documents chain using newer API
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)

    # Full RetrievalQA pipeline
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
        verbose=True
    )

    # Sample queries
    queries = [
        "How many people live in Japan?",
        "How many people live in US?",
        "How many people live in Singapore?"
    ]

    for q in queries:
        result = qa_chain.invoke(q)
        print(f"Q: {q}\nA: {result['result']}\n")


if __name__ == "__main__":
    main()
