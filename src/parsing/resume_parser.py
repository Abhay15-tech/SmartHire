"""Extract text from PDF / DOCX / TXT."""
import pdfplumber
import docx

def extract_text_from_pdf(filepath_or_file):
    text = ""
    with pdfplumber.open(filepath_or_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(filepath_or_file):
    doc = docx.Document(filepath_or_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_txt(filepath_or_file):
    if hasattr(filepath_or_file, "read"):
        content = filepath_or_file.read()
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="ignore")
        return content
    with open(filepath_or_file, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def parse_resume(filepath_or_file, filename):
    """Parse resume based on file extension."""
    ext = str(filename).lower().split('.')[-1]
    if ext == "pdf":
        return extract_text_from_pdf(filepath_or_file)
    elif ext == "docx":
        return extract_text_from_docx(filepath_or_file)
    elif ext == "txt":
        return extract_text_from_txt(filepath_or_file)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
