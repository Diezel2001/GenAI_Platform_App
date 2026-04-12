import pytest
from injestion.loaders import PDFLoader, TextLoader, MarkdownLoader, HTMLLoader, JSONLoader, CSVLoader 

PATH = "injestion/test/data/"

# -------- Tests --------
def test_pdf_loader():
    loader = PDFLoader()
    text = None
    text = loader.load(PATH+"Sample.pdf")
    # print(text)
    assert text is not None
    
def test_text_loader():
    loader = TextLoader()
    text = None
    text = loader.load(PATH + "Sample.txt")
    assert text is not None
    assert isinstance(text, str)
    assert len(text) > 0

def test_markdown_loader():
    loader = MarkdownLoader()
    text = None
    text = loader.load(PATH + "Sample.md")
    assert text is not None
    assert isinstance(text, str)
    assert len(text) > 0

def test_html_loader():
    loader = HTMLLoader()
    text = None
    text = loader.load(PATH + "Sample.html")
    assert text is not None
    assert isinstance(text, str)
    assert len(text) > 0

def test_json_loader():
    loader = JSONLoader()
    text = None
    text = loader.load(PATH + "Sample.json")
    assert text is not None
    assert isinstance(text, str)
    assert len(text) > 0

def test_csv_loader():
    loader = CSVLoader()
    text = None
    text = loader.load(PATH + "Sample.csv")
    assert text is not None
    assert isinstance(text, str)
    assert len(text) > 0
