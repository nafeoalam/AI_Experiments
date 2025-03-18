from langchain.agents import Tool, AgentExecutor
from langchain.agents import initialize_agent, AgentType
from langchain.schema import SystemMessage
from langchain.tools import BaseTool
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import json
import os
from dotenv import load_dotenv
from typing import Optional, Dict, List, Any, Union
from pydantic import Field

# Load environment variables
load_dotenv()   

# Custom Google Search Tool
class CustomGoogleSearchTool(BaseTool):
    name: str = "google_search"
    description: str = "Search Google for RFP information"
    
    api_key: str = Field(description="Google API key")
    custom_search_engine_id: str = Field(description="Google Custom Search Engine ID")
    base_url: str = Field(default="https://www.googleapis.com/customsearch/v1")
    
    def _run(self, query):
        params = {
            "key": self.api_key,
            "cx": self.custom_search_engine_id,
            "q": query,
            "num": 5
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            search_results = response.json()
            
            if "items" not in search_results:
                return "No results found."
                
            formatted_results = []
            
            for i, item in enumerate(search_results["items"][:5]):
                title = item.get("title", "No title")
                link = item.get("link", "No link")
                snippet = item.get("snippet", "No description")
                
                formatted_results.append(f"{i+1}. {title}\nURL: {link}\nDescription: {snippet}\n")
            
            return "\n".join(formatted_results)
        
        except Exception as e:
            return f"An error occurred: {str(e)}"

class WebScraperTool(BaseTool):
    name: str = "web_scraper"
    description: str = "Scrape content from a given URL"
    
    def _run(self, url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text(separator=' ', strip=True)
            return text[:10000]  # Limit length to avoid token issues
        except Exception as e:
            return f"Error scraping {url}: {str(e)}"

class PageNavigatorTool(BaseTool):
    name: str = "page_navigator"
    description: str = "Find the next page link on a paginated website"
    
    def _run(self, url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Common patterns for next page links
            possible_next = soup.select('a.next, .pagination a[rel="next"], a:contains("Next")')
            
            if possible_next:
                next_url = possible_next[0].get('href')
                if next_url:
                    if not next_url.startswith(('http://', 'https://')):
                        next_url = urljoin(url, next_url)
                    return next_url
            
            return "No next page link found"
        except Exception as e:
            return f"Error finding next page: {str(e)}"

class RFPLinkExtractorTool(BaseTool):
    name: str = "rfp_link_extractor"
    description: str = "Extract all RFP links from a page"
    
    def _run(self, url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # This needs to be customized based on the specific site structure
            rfp_links = soup.select('.rfp-listing a, .opportunities a, table.listings a')
            
            links = []
            for link in rfp_links:
                href = link.get('href')
                if href:
                    if not href.startswith(('http://', 'https://')):
                        href = urljoin(url, href)
                    text = link.get_text(strip=True)
                    links.append({"url": href, "text": text})
            
            return links if links else "No RFP links found"
        except Exception as e:
            return f"Error extracting RFP links: {str(e)}"

class KeywordRelevanceAnalyzerTool(BaseTool):
    name: str = "keyword_relevance_analyzer"
    description: str = "Analyze text for relevance to specified keywords"
    
    def _run(self, args):
        try:
            if isinstance(args, str):
                params = json.loads(args)
            else:
                params = args
                
            text = params.get("text", "")
            keywords = params.get("keywords", [])
            
            results = {}
            for keyword in keywords:
                count = text.lower().count(keyword.lower())
                results[keyword] = count
            
            total_occurrences = sum(results.values())
            relevance_score = min(10, (total_occurrences / len(keywords)) * 2) if keywords else 0
            
            return {
                "matches": results,
                "total_occurrences": total_occurrences,
                "relevance_score": relevance_score,
                "relevant": relevance_score > 3  # Adjust threshold as needed
            }
        except Exception as e:
            return f"Error analyzing relevance: {str(e)}"

# Setup the Azure OpenAI LLM
llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    model_name=os.getenv("AZURE_OPENAI_MODEL_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    temperature=0
)

print(os.getenv("GOOGLE_API_KEY"))
print(os.getenv("GOOGLE_CSE_ID"))


# Setup tools with your custom Google search
tools = [
    CustomGoogleSearchTool(
        api_key=os.getenv("GOOGLE_API_KEY"),
        custom_search_engine_id=os.getenv("GOOGLE_CSE_ID")
    ),
    WebScraperTool(),
    PageNavigatorTool(),
    RFPLinkExtractorTool(),
    KeywordRelevanceAnalyzerTool()
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Example execution
def analyze_rfp_site(start_url, keywords):
    response = agent.invoke({
        "input": f"Find relevant RFPs at {start_url} that match my keywords",
        "keywords": keywords
    })
    return response

def load_keywords(file_path="keywords.json"):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            return data.get("keywords", [])
    except Exception as e:
        print(f"Error loading keywords from {file_path}: {str(e)}")
        return ["artificial intelligence", "machine learning", "data science"]  # fallback keywords
        
        
def load_website(file_path="keywords.json"):
    try:
        with open(file_path, "r") as f:     
            data = json.load(f)
            return data.get("website", "")
    except Exception as e:
        print(f"Error loading website from {file_path}: {str(e)}")
        return ""
# Run the agent
if __name__ == "__main__":
    start_url = load_website()# Replace with your actual RFP site
    keywords = load_keywords()
    results = analyze_rfp_site(start_url, keywords)
    print(json.dumps(results, indent=2)) 