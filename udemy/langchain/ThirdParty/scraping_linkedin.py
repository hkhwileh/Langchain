import os
import requests

from dotenv import load_dotenv

load_dotenv()

def scrape_linkedin_profile(linkedin_profile_url:str , mock:bool=False) -> str:
    """
    Scrape a LinkedIn profile and return the content.
    
    Args:
        linkedin_profile (str): The URL of the LinkedIn profile to scrape.
        mock (bool): If True, return a mock response instead of scraping.
        
    Returns:
        str: The scraped content or mock response.
    """
    if mock:
        linkedin_profile_url = "https://gist.githubusercontent.com/emarco177/859ec7d786b45d8e3e3f688c6c9139d8/raw/5eaf8e46dc29a98612c8fe0c774123a7a2ac4575/eden-marco-scrapin.json"
        response = requests.get(linkedin_profile_url,timeout=10)
    else:
        api_endpoint = "https://api.scrapin.io/enrichment/profile"
        params = {
            "url": linkedin_profile_url,
            "api_key": os.getenv("SCRAPIN_API_KEY")
        }

        response = requests.get(linkedin_profile_url, timeout=10,params=params,api_endpoint=api_endpoint)
        data = response.json().get("person")

        data = {
            k:v for k, v in data.items() if v is not ([],"",",None") and k not in ["certifications"]
        }

        return data
    
    # Here you would implement actual scraping logic using libraries like BeautifulSoup or Scrapy.
    # For now, we return a placeholder string.
    return f"Scraped content from {linkedin_profile_url}"


if __name__ == "__main__":
    print("Scraping LinkedIn Profile")

    print (scrape_linkedin_profile("https://www.linkedin.com/in/khwileh", mock=True))