import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def scrape_rfp_pages(base_url, max_pages=10):
    all_rfp_content = []
    for page_num in range(1, max_pages + 1):
        # Handle pagination format
        if page_num == 1:
            url = base_url
        else:
            url = f"{base_url}?page={page_num}"  # Adjust based on actual URL format
        
        response = requests.get(url)
        if response.status_code != 200:
            break
            
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find all RFP links - adjust selector based on actual website structure
        rfp_links = soup.select('.rfp-link a')  # Example selector
        
        for link in rfp_links:
            rfp_url = link['href']
            if not rfp_url.startswith('http'):
                rfp_url = f"{'https://example.com'}/{rfp_url.lstrip('/')}"
            
            rfp_response = requests.get(rfp_url)
            rfp_soup = BeautifulSoup(rfp_response.text, 'html.parser')
            # Extract the main content - adjust selector based on actual website
            content = rfp_soup.select_one('.rfp-content').get_text()  # Example selector
            all_rfp_content.append({
                'url': rfp_url,
                'content': content
            })
    
    return all_rfp_content

def assess_relevance(rfp_data, target_keywords, similarity_threshold=0.3):
    relevant_rfps = []
    
    # Create a vocabulary of target keywords and their variations
    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    
    for rfp in rfp_data:
        # Simple keyword matching
        content = rfp['content'].lower()
        keyword_matches = sum(1 for keyword in target_keywords if keyword.lower() in content)
        
        # More sophisticated semantic similarity
        tfidf = TfidfVectorizer().fit_transform([" ".join(target_keywords), content])
        similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        
        if keyword_matches > 0 or similarity > similarity_threshold:
            relevant_rfps.append({
                'url': rfp['url'],
                'relevance_score': similarity,
                'keyword_matches': keyword_matches
            })
    
    return sorted(relevant_rfps, key=lambda x: (x['keyword_matches'], x['relevance_score']), reverse=True)

# Example usage
base_url = "https://example.com/rfps"
target_keywords = ["artificial intelligence", "machine learning", "data analysis", "predictive modeling"]
rfp_data = scrape_rfp_pages(base_url)
relevant_rfps = assess_relevance(rfp_data, target_keywords)