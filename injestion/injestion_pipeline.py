
import injestion.loaders as loaders
import injestion.chunking as chunking
import injestion.schemas as schemas
import injestion.preprocessing as preprocessing
from typing import List

class PDFDocumentInjestionPipeline:
    def __init__(
        self,
        loader: loaders.PDFLoader,
        chunker: chunking.PDFChunker,
    ):
        self.loader = loader
        self.chunker = chunker

    def run(self, path) -> List[chunking.Chunk]:
        content = self.loader.load(path)
        content_p = preprocessing.normalize_text(content)

        raw_docu = preprocessing.extract_raw_document(path)
        raw_docu.content = content_p

        #chunks = self.chunk_by_page(raw_docu)
        chunks = self.chunk_by_fixed_size(raw_docu)
        #chunks= self.chunk_by_semantics(raw_docu)

        if not chunks:
            return []

        return chunks
