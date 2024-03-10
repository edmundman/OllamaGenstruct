import os
import fitz  # PyMuPDF
import requests
import re
import csv
from html import unescape
from tqdm import tqdm

def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def find_doi(text):
    pattern = r'10.\d{4,9}/[-._;()/:A-Z0-9]+'
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group() if match else None

def fetch_title_from_doi(doi):
    url = f"https://api.crossref.org/works/{doi}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        title = data['message']['title'][0]
        return unescape(re.sub('<[^<]+?>', '', title))  # Remove HTML tags and unescape HTML entities
    except requests.RequestException:
        return None

def split_text_into_chunks(text, chunk_size=1000):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield ' '.join(words[i:i+chunk_size])

def clean_text(text):
    sections_to_remove = ["ACKNOWLEDGMENTS", "REFERENCES", "CITATIONS"]
    for section in sections_to_remove:
        section_index = text.upper().find(section)
        if section_index != -1:
            text = text[:section_index]
    return  unescape(re.sub('<[^<]+?>', '', text))

def process_pdf_folder(folder_path, output_csv):
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Title', 'Content'])

        for filename in tqdm(pdf_files, desc="Processing PDFs"):
            pdf_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(pdf_path)
            text = clean_text(text)  # Clean text to remove acknowledgments, references, citations

            doi = find_doi(text)
            if doi:
                title = fetch_title_from_doi(doi)
                if title:
                    for chunk in split_text_into_chunks(text):
                        writer.writerow([title, chunk])

# Example usage
folder_path = '.'
output_csv = 'output.csv'
process_pdf_folder(folder_path, output_csv)
