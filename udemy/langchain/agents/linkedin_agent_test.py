from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.tools import Tool

from langchain.hub import pull as hub_pull


def lookup(name:str) ->str:
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
    )
    template = """given the full name {name_of_person} I want you to get it me a link to their Linkedin profile page.
                              Your answer should contain only a URL"""
    
    promptTemplate = PromptTemplate(template=template,input_variables =["name_of_person"])

    tools_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func="tools.get_profile_url_tavily",
            description="useful for when you need get the Linkedin Page URL",
        )   
    ]
    react_prompt = hub_pull("hwchase17/react")
    



if __name__ == "__main__":
    print(lookup(name="Eden Marco Udemy"))
    # print(lookup(name="Eden Marco"))
    linkedin_rul = lookup(name='Eden Marco')
    print(linkedin_rul)