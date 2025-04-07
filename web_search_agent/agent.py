import pandas as pd
import openai
import requests
from openai import AzureOpenAI

# Excel extraction
def extract_opportunity_names(excel_path):
    df = pd.read_excel(excel_path)
    if "Opportunity Name" not in df.columns:
        raise ValueError("Excel file must contain an 'Opportunity Name' column")
    return df["Opportunity Name"].tolist()

# Azure Open AI setup
deployment_name = "gpt-4o"

def analyze_opportunities(keywords, opportunity_names):
    prompt = f"""
    You are an agent tasked with finding relevant RFPs based on keywords. Given the keywords: {keywords}, 
    and the following list of Opportunity Names from an RFP Excel file:
    {chr(10).join(opportunity_names)}
    
    Identify which Opportunity Name(s) are most relevant to the keywords. Provide a summary of the matches.
    """
    client = AzureOpenAI(
        api_key="26c38df2f0064988a4c9939d1852acfd",
        api_version="2023-05-15",
        azure_endpoint="https://boh-ai.openai.azure.com/"
    )
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content

# Web search (using SerpAPI)
def search_web(query):
    serpapi_key = "YOUR_SERPAPI_KEY"
    url = "https://serpapi.com/search"
    params = {"q": query, "api_key": serpapi_key}
    response = requests.get(url, params=params)
    results = response.json()
    return results["organic_results"][0]["link"] if results.get("organic_results") else "No link found"

# Main function
def find_relevant_rfp(keywords, excel_path):
    opportunity_names = extract_opportunity_names(excel_path)
    analysis = analyze_opportunities(keywords, opportunity_names)
    print("Analysis Result:\n", analysis)
    
    top_match = None
    # search_query = keywords
    for line in analysis.split("\n"):
        if "Top match:" in line:
            top_match = line.split("Top match:")[-1].strip()
        if "Suggested search query:" in line:
            search_query = line.split("Suggested search query:")[-1].strip()
    
    if not top_match and opportunity_names:
        top_match = opportunity_names[0]
    
    # rfp_link = search_web(search_query)
    return top_match

# Test the agent
keywords = [
    "Application Development Services",
    "IT Consulting",
    "Cyber Security",
    "IT Staffing",
    "Mobile Services",
    "Systems",
    "Software",
    "Network",
    "Chief Medical Examiner"
  ]
excel_path = "RFx Opportunity Report-2025-04-03-13-08-50.xlsx"
top_match = find_relevant_rfp(keywords, excel_path)
print("Top Matching Opportunity Name:", top_match)