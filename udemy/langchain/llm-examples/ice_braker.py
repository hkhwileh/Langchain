from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

if __name__ =='__main__':
    print('Hello Langchain!')
    load_dotenv()
    information='''
        Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman known for his leadership of Tesla, SpaceX, and X (formerly Twitter). Since 2025, he has been a senior advisor to United States president Donald Trump and the de facto head of the Department of Government Efficiency (DOGE). Musk has been considered the wealthiest person in the world since 2021; as of March 2025, Forbes estimates his net worth to be US$345 billion. He was named Time magazine's Person of the Year in 2021.

        Born to a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada. He graduated from the University of Pennsylvania in the U.S. before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. That year, Musk also became a U.S. citizen.


        '''
    
    summary_templete = '''
    given in information {information} from a person i want you to create :
    1. short summary
    2. two interesting facts about them
    '''

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_templete
    )
    llm = ChatOpenAI(temperature=0,model_name='gpt-3.5-turbo')
    chain = summary_prompt_template | llm

    res = chain.invoke(input={'information':information}
                       )
    
    print(res)