import pandas as pd
import requests
from openai import AzureOpenAI
from flask import Flask, request, render_template, redirect, url_for
import os
import docx
import PyPDF2
import re
import httpx

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize Azure OpenAI client once
client = AzureOpenAI(
    api_key="26c38df2f0064988a4c9939d1852acfd",
    api_version="2023-05-15",
    azure_endpoint="https://boh-ai.openai.azure.com/",
    http_client=httpx.Client()  # Create a new client with no proxy
)
deployment_name = "gpt-4o"

# File content extraction
def extract_file_content(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    content = ""
    
    try:
        if file_extension == '.xlsx' or file_extension == '.xls':
            # Excel files
            df = pd.read_excel(file_path)
            # Convert all DataFrame content to string
            content = df.to_string(index=False)
            
        elif file_extension == '.docx':
            # Word documents
            doc = docx.Document(file_path)
            content = "\n".join([para.text for para in doc.paragraphs])
            
        elif file_extension == '.pdf':
            # PDF files - improved extraction
            pdf_content = []
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                # Extract text from all pages
                for page_num in range(total_pages):
                    page_text = pdf_reader.pages[page_num].extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        pdf_content.append(f"--- Page {page_num + 1} ---\n{page_text}")
                
            content = "\n\n".join(pdf_content)
        
        elif file_extension == '.txt':
            # Text files
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
        
        else:
            # For unsupported file types
            content = f"Unsupported file type: {file_extension}. Please upload an Excel, Word, PDF, or text file."
    
    except Exception as e:
        print(f"Error extracting content from {file_path}: {str(e)}")
        content = f"Unable to extract content from file: {str(e)}"
    
    return content

def analyze_file_content(keywords, file_content):
    # Process file in chunks if it's very large
    MAX_CHUNK_SIZE = 5000  # Characters per chunk
    OVERLAP = 500  # Characters of overlap between chunks
    
    # If content is small enough, process it directly
    if len(file_content) <= MAX_CHUNK_SIZE:
        return process_content_chunk(keywords, file_content)
    
    # Otherwise, split into chunks and process each
    chunks = []
    position = 0
    full_analysis = []
    
    # First, create a document summary with keywords
    summary_prompt = f"""
    Analyze the following document content and create a brief summary focused on these keywords: {keywords}.
    
    {file_content[:3000]}
    
    [... Document continues for {len(file_content)} characters total]
    """
    
    # Get document summary
    summary_response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are a document analysis assistant that extracts key information."},
            {"role": "user", "content": summary_prompt}
        ],
        max_tokens=300,
        temperature=0.5
    )
    
    document_summary = summary_response.choices[0].message.content
    full_analysis.append(f"DOCUMENT SUMMARY:\n{document_summary}\n\nDETAILED ANALYSIS:")
    
    # Then process document in chunks
    while position < len(file_content):
        chunk = file_content[position:position + MAX_CHUNK_SIZE]
        
        # Process this chunk
        chunk_result = process_content_chunk(keywords, chunk, position, len(file_content))
        full_analysis.append(chunk_result)
        
        # Move to next chunk with overlap
        position += (MAX_CHUNK_SIZE - OVERLAP)
    
    # Combine all analyses
    combined_analysis = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are a document analysis assistant that extracts key information."},
            {"role": "user", "content": f"""
            I have analyzed a document in chunks. Combine the following analyses into a single coherent summary.
            Focus on the most relevant information related to these keywords: {keywords}.
            
            {' '.join(full_analysis)}
            
            Provide your final combined analysis with the most important findings first. Eliminate repetition.
            """}
        ],
        max_tokens=500,
        temperature=0.5
    )
    
    return combined_analysis.choices[0].message.content

def process_content_chunk(keywords, content_chunk, position=0, total_length=0):
    # Add position context if this is a chunk
    position_info = ""
    if total_length > 0:
        percentage = (position / total_length) * 100
        position_info = f"[Analyzing content from position {position}/{total_length} ({percentage:.1f}%)]"
    
    prompt = f"""
    You are an agent tasked with finding relevant information based on keywords. {position_info}
    
    Given the keywords: {keywords}, analyze the following content:
    
    {content_chunk}
    
    Identify only the parts most relevant to the keywords. Focus on extracting specific information 
    rather than general observations. If nothing relevant is found in this section, state that clearly.
    """
    
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are a precise document analysis tool. Extract only relevant information."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.5
    )
    
    return response.choices[0].message.content

# Main function
def find_relevant_info(keywords, file_path):
    file_content = extract_file_content(file_path)
    analysis = analyze_file_content(keywords, file_content)
    print("Analysis Result:\n", analysis)
    
    # Parse the analysis for matches
    matches = []
    lines = analysis.split("\n")
    for line in lines:
        line = line.strip()
        if line:
            matches.append(line)
    
    return matches, analysis

# Routes
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if file and keywords are provided
        if 'file' not in request.files or 'keywords' not in request.form:
            return "Please upload a file and provide keywords", 400
        
        file = request.files['file']
        keywords_input = request.form['keywords']
        
        if file.filename == '':
            return "No file selected", 400
        
        # Parse comma-separated keywords into a list
        keywords = [kw.strip() for kw in keywords_input.split(',') if kw.strip()]
        if not keywords:
            return "Please provide at least one keyword", 400
        
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        try:
            # Find relevant information
            matches, analysis = find_relevant_info(keywords, file_path)
            # Clean up the uploaded file
            os.remove(file_path)
            # Render the result page
            return render_template('result.html', matches=matches, analysis=analysis)
        except Exception as e:
            os.remove(file_path)
            return f"An error occurred: {str(e)}", 500
    
    return render_template('index.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)