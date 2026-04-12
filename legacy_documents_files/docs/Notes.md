# To dos:

### 1. Large Language Models & Generative AI
- Use GPT-4.1 / Claude / Llama 3 / Mixtral
- Compare **2+ models** on answer quality, latency, and cost
- Track token usage

---

### 2. Transformer Architectures & Embeddings
- Compare embedding models:
  - OpenAI embeddings
  - BGE / Instructor
- Evaluate cosine vs dot-product similarity
- Discuss embedding dimensionality tradeoffs

Add a short write-up:
> *Why this embedding model fits this corpus*

---

### 3. Prompt Engineering & LLM Tuning
- System + user prompt templates
- Few-shot RAG prompting
- Context compression prompts
- Answer verification / self-check prompts

**Bonus:**  
Chain-of-Thought vs structured outputs comparison

---

### 4. Retrieval Pipelines & Vector Indexing
Implement multiple retrieval strategies:
- Semantic search
- Hybrid search (BM25 + vectors)
- Multi-query retrieval
- Re-ranking (cross-encoder or LLM-based)

Measure:
- Recall@k
- MRR
- Latency impact

---

### 5. RAG Architectures & Context Optimization
- Sliding window chunking
- Semantic chunking
- Hierarchical chunking (parent → child)
- Token budget enforcement
- Context pruning

This is where most candidates fall short — doing this well stands out.

---

### 6. Vector Databases & Similarity Search
Use a **real vector database**:
- Pinecone
- Weaviate
- Qdrant

Demonstrate:
- HNSW index configuration
- Metadata filtering (doc type, date, source)
- Namespace or sharding strategy

Explain *why* HNSW was chosen.

---

### 7. LangChain / LlamaIndex
- Use **one as the core framework**
- Optional: mix both
- Implement **custom retrievers**, not just default chains

---

