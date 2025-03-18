import os
from dotenv import load_dotenv
import requests
import json
from openai import AzureOpenAI
import time
from bs4 import BeautifulSoup
import re

# Load environment variables
load_dotenv()

print(os.getenv("AZURE_OPENAI_API_KEY"))
print(os.getenv("AZURE_OPENAI_API_VERSION"))
print(os.getenv("AZURE_OPENAI_ENDPOINT"))

# Initialize Azure OpenAI client instead of standard OpenAI
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Initialize Google Custom Search API credentials
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")  # Custom Search Engine ID

def search_web(query):
    """
    Search the web using Google Custom Search API
    """
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "num": 5  # Number of results
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Search failed with status code {response.status_code}"}

def extract_search_results(search_data):
    """
    Extract relevant information from Google Custom Search results
    """
    results = []
    
    # Extract organic results
    if "items" in search_data:
        for item in search_data["items"]:
            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", "")
            })
    
    return results

def fetch_webpage_content(url):
    """
    Fetch content from a webpage
    """
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # You might want to use BeautifulSoup to parse HTML properly
            # For simplicity, we're just returning the raw text
            return response.text
        else:
            return f"Failed to fetch content: Status code {response.status_code}"
    except Exception as e:
        return f"Error fetching content: {str(e)}"

def run_agent(user_prompt):
    """
    Run the agentic AI to process the user prompt and search the web
    """
    # Step 1: Analyze the user prompt to determine search queries
    system_message = """
    You are a helpful web research assistant. Your goal is to help users find information on the web.
    You can search the web and analyze the results to provide accurate information.
    When given a user question, first determine what search queries would be most effective.
    """
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"I need to find information about: {user_prompt}\n\nWhat search queries should I use to find this information? Provide 1-3 specific search queries."}
    ]
    
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), 
        messages=messages,
        temperature=0.7
    )
    
    search_queries_text = response.choices[0].message.content
    
    # Extract search queries (simple approach - in production you'd want more robust parsing)
    search_queries = [line.strip().strip('"\'') for line in search_queries_text.split('\n') 
                     if line.strip() and not line.startswith("Search query") and not line.startswith("-")]
    search_queries = [q for q in search_queries if len(q) > 5][:3]  # Take up to 3 substantial queries
    
    if not search_queries:
        # Fallback if no queries were properly extracted
        search_queries = [user_prompt]
    
    # Step 2: Execute searches and collect results
    all_results = []
    for query in search_queries:
        print(f"Searching for: {query}")
        search_data = search_web(query)
        results = extract_search_results(search_data)
        all_results.extend(results)
        time.sleep(1)  # Avoid rate limiting
    
    # Step 3: Analyze results and generate a response
    results_text = json.dumps(all_results, indent=2)
    
    analysis_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Original question: {user_prompt}\n\nSearch results:\n{results_text}\n\nBased on these search results, provide a comprehensive answer to the original question. Include relevant facts and cite your sources."}
    ]
    
    final_response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        messages=analysis_messages,
        temperature=0.7
    )
    
    return final_response.choices[0].message.content

def scrape_city_record(keywords, num_pages=3):
    """
    Scrape the NYC City Record Online website for specified keywords
    across multiple paginated pages.
    
    Args:
        keywords (list): List of keywords to search for
        num_pages (int): Number of pages to scrape
        
    Returns:
        dict: Keywords and their matching links
    """
    base_url = "https://a856-cityrecord.nyc.gov/Section"
    results = {keyword: [] for keyword in keywords}
    
    # Start with the main page
    for page in range(1, num_pages + 1):
        print(f"Scraping page {page} of {num_pages}...")
        
        # For the first page, use the base URL; for subsequent pages, add page parameter
        page_url = base_url if page == 1 else f"{base_url}?page={page}"
        
        try:
            html_content = fetch_webpage_content(page_url)
            if "Failed to fetch content" in html_content or "Error fetching content" in html_content:
                print(f"Error accessing page {page}: {html_content}")
                continue
                
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Based on the screenshots and details pages, we need to find the notice titles (h2/h3 elements)
            # Each notice appears to be a block with a title, agency info, and description
            notice_headers = soup.find_all(['h2', 'h3'])
            
            for header in notice_headers:
                # Get the notice title
                title = header.text.strip()
                
                # Find the link to the detail page (might be in the header or a parent element)
                link_elem = header.find('a')
                if not link_elem and header.parent:
                    link_elem = header.parent.find('a')
                
                detail_url = None
                if link_elem and link_elem.get('href'):
                    href = link_elem.get('href')
                    # Make the link absolute if it's relative
                    if href.startswith('/'):
                        detail_url = f"https://a856-cityrecord.nyc.gov{href}"
                    else:
                        detail_url = href
                
                # Try to find the agency info (usually appears after "from")
                agency_text = ""
                agency_elem = None
                sibling = header.next_sibling
                while sibling and not agency_text:
                    if isinstance(sibling, str) and "from" in sibling.lower():
                        agency_text = sibling.strip()
                    elif hasattr(sibling, 'text') and "from" in sibling.text.lower():
                        agency_text = sibling.text.strip()
                    sibling = sibling.next_sibling if hasattr(sibling, 'next_sibling') else None
                
                # If we still don't have agency info, try looking at parent's text
                if not agency_text and header.parent:
                    parent_text = header.parent.text
                    from_index = parent_text.lower().find("from")
                    if from_index > 0:
                        agency_start = from_index + 4  # Length of "from"
                        agency_end = parent_text.find("\n", agency_start)
                        if agency_end > agency_start:
                            agency_text = parent_text[agency_start:agency_end].strip()
                        else:
                            agency_text = parent_text[agency_start:].strip()
                
                # Get text description (could be after the header)
                description = ""
                desc_elem = header.find_next('p')
                if desc_elem:
                    description = desc_elem.text.strip()
                
                # If we have a detail URL, visit that page to get complete information
                detail_info = {}
                if detail_url:
                    print(f"  Visiting detail page: {detail_url}")
                    try:
                        detail_html = fetch_webpage_content(detail_url)
                        if "Failed to fetch" not in detail_html and "Error fetching" not in detail_html:
                            detail_soup = BeautifulSoup(detail_html, 'html.parser')
                            
                            # Extract structured data from the detail page
                            detail_info = extract_detail_page_info(detail_soup)
                            
                            # Update our information with the more detailed data
                            if 'title' in detail_info and detail_info['title']:
                                title = detail_info['title']
                            if 'agency' in detail_info and detail_info['agency']:
                                agency_text = detail_info['agency']
                            if 'description' in detail_info and detail_info['description']:
                                description = detail_info['description']
                        
                        # Pause to avoid overwhelming the server
                        time.sleep(1)
                    except Exception as e:
                        print(f"  Error processing detail page: {str(e)}")
                
                # Combine all text for keyword matching
                all_text = f"{title} {agency_text} {description}"
                
                # Check if this notice matches any of our keywords
                for keyword in keywords:
                    pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                    if pattern.search(all_text) or (detail_url and pattern.search(detail_url)):
                        match_info = {
                            'title': title,
                            'agency': agency_text,
                            'description': description[:200] + '...' if len(description) > 200 else description,
                            'url': detail_url,
                            'page': page
                        }
                        
                        # Add any additional details we found
                        match_info.update({k: v for k, v in detail_info.items() 
                                         if k not in ['title', 'agency', 'description']})
                        
                        results[keyword].append(match_info)
            
            # Sleep to avoid overwhelming the server
            time.sleep(2)
            
        except Exception as e:
            print(f"Error scraping page {page}: {str(e)}")
    
    return results

def extract_detail_page_info(soup):
    """
    Extract structured information from a notice detail page
    
    Args:
        soup: BeautifulSoup object of the detail page
        
    Returns:
        dict: Extracted information
    """
    info = {}
    
    # Extract the title
    title_elem = soup.find('h1')
    if title_elem:
        info['title'] = title_elem.text.strip()
    
    # Look for structured data in key-value pairs
    # Based on the detail page examples, information appears in a structured format
    labels = soup.find_all(['dt', 'strong', 'b']) 
    
    for label in labels:
        label_text = label.text.strip().lower().rstrip(':')
        
        # Find the corresponding value (usually the next sibling or element)
        value = ""
        
        # Try to find the value based on the element type
        if label.name == 'dt':
            # Definition lists: <dt>Label</dt><dd>Value</dd>
            dd = label.find_next('dd')
            if dd:
                value = dd.text.strip()
        else:
            # For other elements, try next sibling or parent's next sibling
            next_elem = label.next_sibling
            if next_elem and isinstance(next_elem, str):
                value = next_elem.strip()
            elif next_elem:
                value = next_elem.text.strip()
        
        # Map common labels to standardized fields
        if label_text in ['agency name', 'agency']:
            info['agency'] = value
        elif label_text in ['description']:
            info['description'] = value
        elif label_text in ['section']:
            info['section'] = value
        elif label_text in ['category']:
            info['category'] = value
        elif label_text in ['selection method']:
            info['selection_method'] = value
        elif label_text in ['vendor information', 'vendor']:
            info['vendor'] = value
        elif label_text in ['contract amount']:
            info['contract_amount'] = value
        elif label_text in ['pin']:
            info['pin'] = value
        elif label_text in ['publication date']:
            info['publication_date'] = value
        elif label_text in ['notice type']:
            info['notice_type'] = value
        elif label_text in ['contact information', 'contact']:
            info['contact'] = value
        else:
            # Store other fields with their original label
            clean_label = label_text.replace(' ', '_')
            info[clean_label] = value
    
    # If we couldn't find a structured description, try to extract it from a paragraph
    if 'description' not in info:
        # Look for a paragraph that might contain the description
        description_elem = soup.find('p')
        if description_elem:
            info['description'] = description_elem.text.strip()
    
    return info

def main():
    print("NYC City Record Scraper")
    print("----------------------")
    
    # Import keywords from JSON file
    try:
        with open('keywords.json', 'r') as file:
            keywords_data = json.load(file)
            keywords = keywords_data.get('keywords', [])
            
        if not keywords:
            print("Error: No keywords found in the JSON file. Please check the format.")
            return
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading keywords from JSON: {str(e)}")
        print("Please make sure 'keywords.json' exists and contains valid JSON.")
        return
    
    # Scrape the website
    print(f"Scraping the NYC City Record website for {len(keywords)} keywords...")
    print(f"Keywords: {', '.join(keywords)}")
    results = scrape_city_record(keywords, num_pages=3)
    
    # Output the results
    print("\nResults:")
    print("-" * 50)
    
    found_any = False
    for keyword, matches in results.items():
        if matches:
            found_any = True
            print(f"\n{keyword} ({len(matches)} matches):")
            for i, match in enumerate(matches, 1):
                print(f"{i}. {match.get('title', 'No Title')}")
                if match.get('agency'):
                    print(f"   Agency: {match['agency']}")
                if match.get('vendor'):
                    print(f"   Vendor: {match['vendor']}")
                if match.get('contract_amount'):
                    print(f"   Contract Amount: {match['contract_amount']}")
                if match.get('description'):
                    print(f"   Description: {match['description']}")
                print(f"   URL: {match['url']}")
                print(f"   Page: {match['page']}")
                print()
    
    if not found_any:
        print("No matches found for the specified keywords.")
    
    # Export results to JSON file for easier analysis
    with open('scraping_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results exported to 'scraping_results.json'")

if __name__ == "__main__":
    main()