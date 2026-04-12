CREATE EXTENSION IF NOT EXISTS vector;

-- users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name TEXT,
    email TEXT UNIQUE
);

-- documents table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    title TEXT,
    created_at TIMESTAMP DEFAULT now()
);

-- chunks table (with vector column)
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    document_id INT REFERENCES documents(id),
    content TEXT,
    embedding vector(1536) -- your embedding dimension
);

-- queries table
CREATE TABLE queries (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    query TEXT,
    created_at TIMESTAMP DEFAULT now()
);

-- usage_logs table
CREATE TABLE usage_logs (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    action TEXT,
    created_at TIMESTAMP DEFAULT now()
);

CREATE INDEX ON chunks
USING hnsw (embedding vector_cosine_ops);