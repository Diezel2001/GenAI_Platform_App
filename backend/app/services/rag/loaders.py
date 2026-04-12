from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path

class Loader(ABC):
    @abstractmethod
    def load(self, source):
        pass


from pypdf import PdfReader
class PDFLoader(Loader):
    def load(self, source):
        text = ""
        # Handle both file paths (str/Path) and in-memory bytes/BytesIO
        if isinstance(source, (bytes, bytearray)):
            if len(source) == 0:
                raise ValueError("Cannot read an empty file: source bytes are empty")
            reader = PdfReader(BytesIO(source))
        elif isinstance(source, BytesIO):
            source.seek(0, 2)  # Seek to end
            size = source.tell()
            source.seek(0)  # Reset to beginning
            if size == 0:
                raise ValueError("Cannot read an empty file: BytesIO is empty")
            reader = PdfReader(source)
        elif isinstance(source, (str, Path)):
            import os
            if os.path.getsize(source) == 0:
                raise ValueError(f"Cannot read an empty file: {source}")
            reader = PdfReader(source)
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")
        
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

# class TextLoader(Loader):
#     def load(self, source):
#         with open(source, "r", encoding="utf-8") as f:
#             return f.read()

# class MarkdownLoader(Loader):
#     def load(self, source):
#         with open(source, "r", encoding="utf-8") as f:
#             return f.read()
        
# import csv
# class CSVLoader(Loader):
#     def load(self, source):
#         rows = []
#         with open(source, newline="", encoding="utf-8") as f:
#             reader = csv.reader(f)
#             for row in reader:
#                 rows.append(", ".join(row))
#         return "\n".join(rows)

# import json
# class JSONLoader(Loader):
#     def load(self, source):
#         with open(source, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         return json.dumps(data, indent=2, ensure_ascii=False)


# from bs4 import BeautifulSoup
# class HTMLLoader(Loader):
#     def load(self, source):
#         with open(source, "r", encoding="utf-8") as f:
#             soup = BeautifulSoup(f, "html.parser")
#             return soup.get_text(separator="\n")