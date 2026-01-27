# RAG Knowledge Assistant – Features, Epics, and Stories

- [X] Initialize project

## Feature 1: Document Ingestion & Chunking
**Goal:** Load, clean, and split documents into manageable chunks.

### Epic 1.1: Raw Data Loaders
- [X] Story 1.1.1: Implement PDF loader
- [X] Story 1.1.2: Implement Markdown loader
- [X] Story 1.1.3: Implement HTML loader
- [ ] Story 1.1.4: Implement Git repository loader   
- [X] Story 1.1.5: unit testing

extras:
- [X] Implement JSON, txt, csv, loader

### Epic 1.2: Preprocessing & Cleaning
- [X] Story 1.2.1: Normalize text
- [X] Story 1.2.2: Remove boilerplate / navigation text
- [X] Story 1.2.3: Extract metadata (date, source, type)
- [X] Story 1.2.4: unit testing

extras:
- [X] raw document input mechanism

### Epic 1.3: Chunking Strategies
- [X] Story 1.3.1: Fixed-size chunking, page chunking, header chunking
- [ ] Story 1.3.2: Semantic chunking
- [ ] Story 1.3.3: Hierarchical chunking (parent-child)
- [X] Story 1.3.4: unit testing

### Epic 1.4: Document Schema
- [X] Story 1.4.1: Define Document & Chunk schemas
- [X] Story 1.4.2: Store chunk metadata for retrieval filtering

---

## Feature 2: Embedding Generation
**Goal:** Convert text chunks into vector embeddings.

### Epic 2.1: Embedding Models
- [X] Story 2.1.1: Integrate OpenAI embeddings
- [X] Story 2.1.2: Integrate BGE / Instructor embeddings

### Epic 2.2: Embedding Pipeline
- [ ] Story 2.2.1: Batch embedding with retries
- [ ] Story 2.2.3: Save embeddings to vector store

### Epic 2.3: Similarity & Evaluation
- [ ] Story 2.3.1: Implement cosine similarity
- [ ] Story 2.3.2: Implement dot-product similarity
- [ ] Story 2.3.3: Compare embedding dimensions & effect on recall

---

## Feature 3: Vector Database & Indexing
**Goal:** Store and retrieve embeddings efficiently.

### Epic 3.1: Vector DB Integration
- [ ] Story 3.1.1: Connect to Faiss / Qdrant / Pinecone / Weaviate
- [ ] Story 3.1.2: Set up HNSW index

### Epic 3.2: Metadata Filtering
- [ ] Story 3.2.1: Implement filtering by document type, date, or source

### Epic 3.3: Sharding & Namespaces
- [ ] Story 3.3.1: Implement namespace-based sharding
- [ ] Story 3.3.2: Test retrieval across multiple shards

---

## Feature 4: Retrieval Pipelines
**Goal:** Fetch the most relevant documents for a query.

### Epic 4.1: Basic Retrieval
- [ ] Story 4.1.1: Semantic-only vector search
- [ ] Story 4.1.2: Evaluate recall@k

### Epic 4.2: Hybrid Retrieval
- [ ] Story 4.2.1: Integrate BM25 + vector search
- [ ] Story 4.2.2: Evaluate hybrid performance

### Epic 4.3: Multi-query & Rewriting
- [ ] Story 4.3.1: Implement query rewriting via LLM
- [ ] Story 4.3.2: Test multi-hop retrieval

### Epic 4.4: Re-ranking
- [ ] Story 4.4.1: Implement cross-encoder reranking
- [ ] Story 4.4.2: Implement LLM-based reranking
- [ ] Story 4.4.3: Evaluate precision vs latency

---

## Feature 5: Context Optimization
**Goal:** Fit retrieved chunks into LLM token limits while maximizing answer quality.

### Epic 5.1: Token Budget Management
- [ ] Story 5.1.1: Assemble context based on max token budget

### Epic 5.2: Context Compression
- [ ] Story 5.2.1: Summarize chunks for token reduction
- [ ] Story 5.2.2: Deduplicate overlapping content

### Epic 5.3: Windowing & Sliding Context
- [ ] Story 5.3.1: Implement sliding window retrieval

---

## Feature 6: LLM Prompting
**Goal:** Generate answers with structured prompts.

### Epic 6.1: Prompt Templates
- [ ] Story 6.1.1: System prompt
- [ ] Story 6.1.2: RAG answer prompt
- [ ] Story 6.1.3: Verification/self-check prompt

### Epic 6.2: Prompt Experiments
- [ ] Story 6.2.1: Test few-shot vs zero-shot
- [ ] Story 6.2.2: Compare Chain-of-Thought vs structured outputs

---

## Feature 7: Evaluation & Benchmarking
**Goal:** Measure retrieval and answer quality quantitatively.

### Epic 7.1: Retrieval Metrics
- [ ] Story 7.1.1: Recall@k
- [ ] Story 7.1.2: Mean Reciprocal Rank (MRR)

### Epic 7.2: Answer Quality
- [ ] Story 7.2.1: Faithfulness metric
- [ ] Story 7.2.2: Hallucination detection

### Epic 7.3: Benchmark Comparisons
- [ ] Story 7.3.1: Compare chunking strategies
- [ ] Story 7.3.2: Compare retrieval strategies
- [ ] Story 7.3.3: Compare embedding models

---

## Feature 8: API & UI (Demo)
**Goal:** Expose system for interactive demos.

### Epic 8.1: API
- [ ] Story 8.1.1: Build FastAPI endpoints for queries
- [ ] Story 8.1.2: Return retrieved chunks + LLM answer + sources

### Epic 8.2: UI
- [ ] Story 8.2.1: Simple web app / Streamlit
- [ ] Story 8.2.2: Strategy selector (semantic/hybrid/re-ranked)
- [ ] Story 8.2.3: Display chunk citations

---

## Feature 9: Tests & Notebooks
**Goal:** Ensure correctness and provide research insights.

### Epic 9.1: Unit & Integration Tests
- [ ] Story 9.1.1: Test chunking and embedding pipelines
- [ ] Story 9.1.2: Test retrieval and reranking

### Epic 9.2: Experimentation Notebooks
- [ ] Story 9.2.1: Chunking strategy experiments
- [ ] Story 9.2.2: Embedding visualization
- [ ] Story 9.2.3: Failure case analysis
