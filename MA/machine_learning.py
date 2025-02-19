# machine_learning.py

import PyPDF2
import os
from io import BytesIO

def extract_pdf_text(file):
    """Extract text from a PDF file."""
    if isinstance(file, BytesIO):
        reader = PyPDF2.PdfReader(file)
    else:
        reader = PyPDF2.PdfReader(file)

    text = ""
    for page in reader.pages:
        text += page.extract_text()

    return text

def extract_pdf_from_local(file_path):
    """Extract text from a PDF file located locally."""
    with open(file_path, 'rb') as file:
        return extract_pdf_text(file)
