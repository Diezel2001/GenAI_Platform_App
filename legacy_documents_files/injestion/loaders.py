from abc import ABC, abstractmethod

class Loader(ABC):
    @abstractmethod
    def load(self, source):
        pass


from pypdf import PdfReader
class PDFLoader(Loader):
    def load(self, source):
        text = ""
        reader = PdfReader(source)
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

class TextLoader(Loader):
    def load(self, source):
        with open(source, "r", encoding="utf-8") as f:
            return f.read()

class MarkdownLoader(Loader):
    def load(self, source):
        with open(source, "r", encoding="utf-8") as f:
            return f.read()
        
import csv
class CSVLoader(Loader):
    def load(self, source):
        rows = []
        with open(source, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(", ".join(row))
        return "\n".join(rows)

import json
class JSONLoader(Loader):
    def load(self, source):
        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, indent=2, ensure_ascii=False)


from bs4 import BeautifulSoup
class HTMLLoader(Loader):
    def load(self, source):
        with open(source, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            return soup.get_text(separator="\n")