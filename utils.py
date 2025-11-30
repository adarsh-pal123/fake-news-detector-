# utils.py
import re
import string

def clean_text(text):
    """Simple text cleaning"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)   # remove urls
    text = re.sub(r'[^a-z\s]', ' ', text)           # allow letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()        # normalize whitespace
    return text
