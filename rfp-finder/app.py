import pandas as pd
import requests
from openai import AzureOpenAI
from flask import Flask, request, render_template, redirect, url_for
import os

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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
    
    # Parse the analysis for multiple matches
    matches = []
    lines = analysis.split("\n")
    for line in lines:
        line = line.strip()
        matches.append(line)
    
    return matches, analysis

# # Test the agent
# keywords = [
#     "Application Development Services",
#     "IT Consulting",
#     "Cyber Security",
#     "IT Staffing",
#     "Mobile Services",
#     "Systems",
#     "Software",
#     "Network",
#     "Chief Medical Examiner"
#   ]
# excel_path = "RFx Opportunity Report-2025-04-03-13-08-50.xlsx"
# top_match = find_relevant_rfp(keywords, excel_path)
# print("Top Matching Opportunity Name:", top_match)

# Routes
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if file and keywords are provided
        if 'file' not in request.files or 'keywords' not in request.form:
            return "Please upload an Excel file and provide keywords", 400
        
        file = request.files['file']
        keywords_input = request.form['keywords']
        
        if file.filename == '':
            return "No file selected", 400
        
        # Parse comma-separated keywords into a list
        keywords = [kw.strip() for kw in keywords_input.split(',') if kw.strip()]
        if not keywords:
            return "Please provide at least one keyword", 400
        
        # Save the uploaded file
        excel_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(excel_path)
        
        try:
            # Find relevant RFP
            matches, analysis = find_relevant_rfp(keywords, excel_path)
            # Clean up the uploaded file
            os.remove(excel_path)
            # Render the result page
            return render_template('result.html', matches=matches, analysis=analysis)
        except Exception as e:
            os.remove(excel_path)
            return f"An error occurred: {str(e)}", 500
    
    return render_template('index.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)