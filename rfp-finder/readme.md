# RFP Finder

RFP Finder is a web application that allows users to upload documents (PDF, Word, Excel, or text files) and extract relevant information based on specified keywords. The application uses Azure OpenAI to analyze document content and identify sections matching the given keywords.

## Features

- Upload various document types (PDF, DOCX, XLSX, TXT)
- Analyze documents using Azure OpenAI's GPT-4o model
- Extract relevant sections based on user-defined keywords
- Process documents in chunks for efficient analysis of large files
- Clean, responsive web interface

## Installation

### Prerequisites

- Python 3.12 or later
- Git

### Setup Instructions

1. Clone the repository:
   ```
   git clone https://svamint.visualstudio.com/SVAM_AI/_git/SVAM_AI_ML_POCS
   cd SVAM_AI_ML_POCS/rfp-finder
   ```

2. Create and activate a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up the Azure OpenAI configuration:
   Update the API key and endpoint in the `app.py` file or set environment variables.

5. Create an uploads directory:
   ```
   mkdir -p uploads
   ```

## Running the Application

1. Ensure your virtual environment is activated:
   ```
   source venv/bin/activate
   ```

2. Start the Flask application:
   ```
   python app.py
   ```

3. Open a web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

## Usage

1. Upload a document (PDF, Word, Excel, or text file)
2. Enter keywords separated by commas
3. Click "Find RFP" to analyze the document
4. View the extracted relevant sections

## Dependencies

- Flask: Web framework
- Pandas: Data processing
- OpenPyXL: Excel file support
- Python-DOCX: Word document processing
- PyPDF2: PDF file processing
- Azure OpenAI: AI-powered document analysis

## Development

### Setting up Git Identity

To contribute to this repository, set up your Git identity:

```bash
git config --global user.email "your-email@example.com"
git config --global user.name "Your Name"
```

### Using Azure DevOps with PAT

1. Create a Personal Access Token in Azure DevOps
2. Configure your remote URL:
   ```
   git remote set-url origin https://{username}:{PAT}@svamint.visualstudio.com/SVAM_AI/_git/SVAM_AI_ML_POCS
   ```

## License

Proprietary - SVAM International Inc.
