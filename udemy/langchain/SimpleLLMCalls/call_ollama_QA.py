from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
import os


if __name__ == "__main__":
    print("Hello Ollama World")
    template = "You are an analyst to analyze the tweets and help me to classify them as positive or negative. {tweet}"
    prompt = PromptTemplate.from_template(template)

    llms = ChatOllama(temperature=0.1, model="mixtral")
    chain = prompt | llms | StrOutputParser()
    res = chain.invoke(input={"tweet": "I love the new features of this product!"})
    print(res)

