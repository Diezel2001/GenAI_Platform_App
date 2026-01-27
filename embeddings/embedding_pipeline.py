import models
import injestion.loaders as loaders
import injestion.chunking as chunking
import injestion.schemas as schemas
import injestion.preprocessing as schemas
from typing import List

class PDFDocumentEmbeddingPipeline:
    def __init__(
        self,
        embedding_model: models.EmbeddingModel,
        chunker: chunking.PDFChunker,
    ):
        self.embedding_model = embedding_model
        self.chunker = chunker

    def run(self, chunks) -> List[schemas.RecordData]:


        texts = [c.text for c in chunks]

        embeddings = self.embedding_model.embed_documents(texts)

        
        records: List[schemas.RecordData] = []
        for chunk, embedding in zip(chunks, embeddings):
            records.append(
                schemas.RecordData(
                    id=chunk.id,
                    text=chunk.text,
                    embedding=embedding,
                    metadata=chunk.metadata,
                )
            )

        return records
