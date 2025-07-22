from backend.core import run_llm
import streamlit as st

def create_source_string(source_urls: Set[str])->str:
    if not source_urls:
        return "nothing to display"
    
    source_list = list(source_urls)
    source_list.sort()
    source_string = "source\n"
    for i , source in enumerate(source_list):
        source_string += f"{i+1}. {source}\n"
    return source_string

st.header("langchain Hassan Course Documentation Helper Bot")

prompt = st.text_input("Prompt",placeholder="Enter you prompt here ...")

if prompt:
    with st.spinner("generating response.."):
        generated_response = run_llm(prompt)
        sources = set([doc.metadata["source"] for doc in generated_response["source_documents"]])

        formatted_response = f"{generated_response["result"]} \n\n {create_source_string(sources)}"
