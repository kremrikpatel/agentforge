-- Initial pgvector setup for long-term memory.
-- Runs once, on an empty Postgres volume. memory/ltm.py:SCHEMA mirrors this so a
-- database provisioned some other way still converges to the same shape.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS ltm_records (
    id          UUID PRIMARY KEY,
    session_id  TEXT NOT NULL,
    run_id      TEXT NOT NULL,
    topic       TEXT NOT NULL,
    stage       TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(384),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ltm_records_session_idx ON ltm_records (session_id);

-- HNSW with cosine distance: recall() orders by `embedding <=> query`.
CREATE INDEX IF NOT EXISTS ltm_records_embedding_idx
    ON ltm_records USING hnsw (embedding vector_cosine_ops);
