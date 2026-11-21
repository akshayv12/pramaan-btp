import pymupdf as fitz  # PyMuPDF
from bs4 import BeautifulSoup
from bs4.dammit import EncodingDetector
import os

def extract_text_direct(file_path: str, content_type=None) -> str:
    """
    Extracts text directly from HTML or text-based PDFs.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if content_type == "html" or ext in ['.html', '.htm']:
        with open(file_path, 'rb') as f:
            data = f.read()
            try:
                markup = data.decode('utf-8-sig')
            except UnicodeDecodeError:
                charset = EncodingDetector.find_declared_encoding(data, is_html=True) or 'utf-8'
                try:
                    markup = data.decode(charset, errors='replace')
                except LookupError:
                    markup = data.decode('utf-8', errors='replace')
            soup = BeautifulSoup(markup, 'html.parser')
            
            # Prefer article content over page chrome. Keep the existing direct
            # extraction route and preserve body text verbatim apart from spacing.
            article_nodes = soup.select('.mw-parser-output')
            root = max(article_nodes, key=lambda node: len(node.get_text()), default=None) or soup.find('article') or soup.find('main') or soup.body or soup
            junk_tags = ['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript', 'form', 'button', 'select', 'input']
            for element in root(junk_tags):
                element.decompose()
            for element in root.select('[role="navigation"], [role="complementary"], .navbox, .vertical-navbox, .sidebar, .infobox, .toc, .mw-editsection, .reflist, .catlinks, .metadata, .reference'):
                element.decompose()
            # Inline links/spans must not become sentence/paragraph boundaries.
            for element in root.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li', 'blockquote', 'tr']):
                element.insert_before('\n')
                element.insert_after('\n')
            for element in root.find_all('br'):
                element.replace_with('\n')
            text = root.get_text(separator=' ')
            lines = [' '.join(line.split()) for line in text.split('\n') if line.strip()]
            return '\n'.join(lines)
            
    elif ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
            
    elif ext == '.pdf':
        try:
            doc = fitz.open(file_path)
            full_text = []
            for page in doc:
                full_text.append(page.get_text())
            return '\n'.join(full_text)
        except Exception as e:
            print(f"Error extracting PDF directly: {e}")
            return ""
            
    else:
        print(f"Unsupported file format for direct extraction: {ext}")
        return ""
