

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

"""
This script demonstrates how to create a simple prompt using LangChain and
 OpenAI's GPT model.
using LangChain's PromptTemplate and ChatOpenAI classes, we can create a question answer system
"""


if __name__ == "__main__":
    print("Hello World")

    summary_template = "You are dubai courts judge and you need to answer {question} in arabic legal questions"

    prompt = PromptTemplate(input_variables="question",template=summary_template)

    llms = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")

    chain = prompt | llms

    res = chain.invoke(input={"question":"what is if i give cheque without balance ? "})

    print(res)