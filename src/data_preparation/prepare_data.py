import os
import re
import glob
from pathlib import Path
from typing import List
import pandas as pd
import PyPDF2
import docx

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text


def extract_text_from_docx(docx_path: str) -> str:
    doc = docx.Document(docx_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text


def extract_text_from_txt(txt_path: str) -> str:
    with open(txt_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def clean_text(text: str) -> str:
    # Entferne unerwünschte Zeichen und Formatierungen
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    # Weitere Bereinigungen können hier hinzugefügt werden
    return text.strip()


def prepare_data(raw_data_dir: str, processed_data_dir: str):
    # Erstelle den Ausgabeordner, falls nicht vorhanden
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Unterstützte Dateitypen
    supported_formats = ['*.pdf', '*.docx', '*.txt']
    
    # Sammle alle Dateien
    files = []
    for fmt in supported_formats:
        files.extend(glob.glob(os.path.join(raw_data_dir, fmt)))
    
    data_entries = []
    
    for file_path in files:
        if file_path.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif file_path.endswith('.docx'):
            text = extract_text_from_docx(file_path)
        elif file_path.endswith('.txt'):
            text = extract_text_from_txt(file_path)
        else:
            continue  # Nicht unterstütztes Format
        
        cleaned_text = clean_text(text)
        
        # Optional: Segmentierung des Textes in Abschnitte
        # Hier könntest du den Text in kleinere Teile aufteilen
        
        data_entries.append({
            'file_path': file_path,
            'content': cleaned_text
        })
    
    # Speichere die verarbeiteten Daten, z.B. als CSV oder JSON
    df = pd.DataFrame(data_entries)
    df.to_csv(os.path.join(processed_data_dir, 'processed_data.csv'), index=False)


if __name__ == '__main__':
    raw_data_dir = 'data/raw'
    processed_data_dir = 'data/processed'
    prepare_data(raw_data_dir, processed_data_dir)
