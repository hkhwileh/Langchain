from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import pandas as pd
import os

def extract_sentiment_from_tweet(tweet_text):
    information = '''
        Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman known for his leadership of Tesla, SpaceX, and X (formerly Twitter). Since 2025, he has been a senior advisor to United States president Donald Trump and the de facto head of the Department of Government Efficiency (DOGE). Musk has been considered the wealthiest person in the world since 2021; as of March 2025, Forbes estimates his net worth to be US$345 billion. He was named Time magazine's Person of the Year in 2021.
        Born to a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada. He graduated from the University of Pennsylvania in the U.S. before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. That year, Musk also became a U.S. citizen.
    '''



    summary_templete = '''
    given in tweet from twitter {information}, you have to perform sentement analysis and tell me if the tweet is positive, negative or neutral.'''

    summary_template_prompt= PromptTemplate(input=["information"])

    llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')

    chain = summary_template_prompt | llm
    res = chain.invoke(inpput={'information':information})
    return res

if __name__ == '__main__':
    print('Hello Langchain!')

    #data_sheet = pd.read_csv(r'/Users/dc-hassan/Desktop/AI/dataset/tweets/posts.csv')

    csv_path = '/Users/dc-hassan/Desktop/AI/dataset/tweets/posts.csv'
    if os.path.exists(csv_path):
        data_sheet = pd.read_csv(csv_path)
    else:
        print("❌ File not found:", csv_path)
    data_sheet = pd.read_csv(csv_path)

    
    data_sheet = data_sheet.dropna()
    data_sheet = data_sheet.drop_duplicates()

    data_sheet = data_sheet.reset_index(drop=True)
    first_five = data_sheet['description'].head(5)
    data_sheet_descriptions_only = data_sheet['description'].tolist()
    
    for description in data_sheet_descriptions_only:
        sentiment = extract_sentiment_from_tweet(description)
        print(f"Tweet: {description}\nSentiment: {sentiment}\n")
    print("First five descriptions:")
    print(first_five)
    print(print(data_sheet.columns))



