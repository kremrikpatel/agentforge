"""Chunking, Qdrant ingestion, and RRF-fused hybrid search."""

from __future__ import annotations

import re

from qdrant_client import AsyncQdrantClient, models

from app.config import Settings, get_settings
from app.observability import get_logger, log_event, timed
from memory.embedding import embed
from rag.cache import RagCache
from rag.config import RagSettings, get_rag_settings
from rag.lexical import InMemoryBM25, LexicalIndex
from rag.schemas import Chunk, Document, ScoredChunk

logger = get_logger("agentforge.rag.store")

_PARA = re.compile(r"\n\s*\n")


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def chunk_document(doc: Document, kb_id: str, size: int, overlap: int) -> list[Chunk]:
    """Pack whole paragraphs up to `size`, carrying `overlap` chars between chunks.

    Splitting on paragraph boundaries keeps a chunk semantically whole; the
    overlap stops an answer that straddles a boundary from being lost by both.
    """
    text = doc.text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in _PARA.split(text) if p.strip()]
    pieces: list[str] = []
    for para in paragraphs:
        # A single paragraph longer than the window gets hard-split.
        if len(para) > size:
            stride = size - overlap if size > overlap else size
            for start in range(0, len(para), stride):
                pieces.append(para[start : start + size])
        else:
            pieces.append(para)

    packed: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > size:
            packed.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{piece}".strip() if tail else piece
        else:
            current = f"{current}\n\n{piece}".strip() if current else piece
    if current:
        packed.append(current)

    return [
        Chunk(
            id=Chunk.make_id(kb_id, doc.doc_id, i),
            kb_id=kb_id,
            doc_id=doc.doc_id,
            ordinal=i,
            text=body,
            title=doc.title,
            source=doc.source,
            metadata=doc.metadata,
        )
        for i, body in enumerate(packed)
    ]


# --------------------------------------------------------------------------
# Vector store
# --------------------------------------------------------------------------


class VectorStore:
    """Qdrant dense side. Defaults to embedded local mode -- no server needed."""

    def __init__(
        self,
        rag_settings: RagSettings | None = None,
        settings: Settings | None = None,
        client: AsyncQdrantClient | None = None,
        cache: RagCache | None = None,
    ) -> None:
        self.rag = rag_settings or get_rag_settings()
        self.settings = settings or get_settings()
        self._client = client
        self.cache = cache
        self._ready: set[str] = set()

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            if self.rag.qdrant_url:
                self._client = AsyncQdrantClient(
                    url=self.rag.qdrant_url, api_key=self.rag.qdrant_api_key or None
                )
            else:
                self._client = AsyncQdrantClient(location=self.rag.qdrant_path or ":memory:")
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def ensure_collection(self, kb_id: str) -> None:
        name = self.rag.collection(kb_id)
        if name in self._ready:
            return
        if not await self.client.collection_exists(name):
            await self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=self.rag.embedding_dim, distance=models.Distance.COSINE
                ),
            )
        self._ready.add(name)

    async def embed_text(self, text: str) -> list[float]:
        if self.cache is not None:
            cached = await self.cache.get_embedding(text)
            if cached is not None and len(cached) == self.rag.embedding_dim:
                return cached
        vector = embed(text, self.rag.embedding_dim)
        if self.cache is not None:
            await self.cache.set_embedding(text, vector)
        return vector

    async def upsert(self, kb_id: str, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        await self.ensure_collection(kb_id)
        points = [
            models.PointStruct(
                id=c.id,
                vector=await self.embed_text(f"{c.title}\n{c.text}".strip()),
                payload=c.model_dump(mode="json"),
            )
            for c in chunks
        ]
        await self.client.upsert(collection_name=self.rag.collection(kb_id), points=points)
        return len(points)

    async def dense_search(
        self, kb_id: str, vector: list[float], limit: int
    ) -> list[tuple[Chunk, float]]:
        await self.ensure_collection(kb_id)
        resp = await self.client.query_points(
            collection_name=self.rag.collection(kb_id),
            query=vector,
            limit=limit,
            with_payload=True,
        )
        return [(Chunk.model_validate(p.payload), float(p.score)) for p in resp.points]

    async def hydrate(self, kb_id: str, ids: list[str]) -> dict[str, Chunk]:
        if not ids:
            return {}
        await self.ensure_collection(kb_id)
        records = await self.client.retrieve(
            collection_name=self.rag.collection(kb_id), ids=ids, with_payload=True
        )
        return {str(r.id): Chunk.model_validate(r.payload) for r in records}

    async def drop(self, kb_id: str) -> None:
        name = self.rag.collection(kb_id)
        if await self.client.collection_exists(name):
            await self.client.delete_collection(name)
        self._ready.discard(name)


# --------------------------------------------------------------------------
# Hybrid retrieval
# --------------------------------------------------------------------------


def reciprocal_rank_fusion(ranked_lists: dict[str, list[str]], k: int) -> dict[str, float]:
    """RRF: sum of 1/(k + rank) across every list an id appears in.

    Rank-based rather than score-based, which is the point -- dense cosine and
    BM25 scores are on incomparable scales and cannot be added directly.
    """
    fused: dict[str, float] = {}
    for ids in ranked_lists.values():
        for rank, doc_id in enumerate(ids, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore | None = None,
        lexical: LexicalIndex | None = None,
        rag_settings: RagSettings | None = None,
    ) -> None:
        self.rag = rag_settings or get_rag_settings()
        self.store = store or VectorStore(self.rag)
        self.lexical = lexical if lexical is not None else InMemoryBM25()

    async def ingest(self, kb_id: str, documents: list[Document]) -> int:
        chunks: list[Chunk] = []
        for doc in documents:
            chunks.extend(
                chunk_document(doc, kb_id, self.rag.chunk_size, self.rag.chunk_overlap)
            )
        with timed() as t:
            written = await self.store.upsert(kb_id, chunks)
            await self.lexical.index(kb_id, chunks)
        log_event(
            logger,
            "rag.ingested",
            kb_id=kb_id,
            documents=len(documents),
            chunks=written,
            latency_ms=t["ms"],
        )
        return written

    async def search(
        self,
        kb_id: str,
        *,
        dense_query: str,
        lexical_query: str,
        use_dense: bool = True,
        use_lexical: bool = True,
        limit: int | None = None,
    ) -> list[ScoredChunk]:
        """`dense_query` and `lexical_query` differ under HyDE -- the hypothetical
        document is a good embedding target but poor BM25 input.
        """
        limit = limit or self.rag.top_k_fused

        dense: list[tuple[Chunk, float]] = []
        lexical: list[tuple[str, float]] = []

        if use_dense:
            vector = await self.store.embed_text(dense_query)
            dense = await self.store.dense_search(kb_id, vector, self.rag.top_k_dense)
        if use_lexical:
            lexical = await self.lexical.search(kb_id, lexical_query, self.rag.top_k_lexical)

        # Single-signal modes skip fusion; there is nothing to fuse.
        if use_dense and not use_lexical:
            return [
                ScoredChunk(chunk=c, score=s, dense_score=s, dense_rank=i)
                for i, (c, s) in enumerate(dense[:limit], start=1)
            ]
        if use_lexical and not use_dense:
            hydrated = await self.store.hydrate(kb_id, [cid for cid, _ in lexical[:limit]])
            return [
                ScoredChunk(chunk=hydrated[cid], score=s, lexical_score=s, lexical_rank=i)
                for i, (cid, s) in enumerate(lexical[:limit], start=1)
                if cid in hydrated
            ]

        dense_ids = [c.id for c, _ in dense]
        lexical_ids = [cid for cid, _ in lexical]
        fused = reciprocal_rank_fusion(
            {"dense": dense_ids, "lexical": lexical_ids}, self.rag.rrf_k
        )
        if not fused:
            return []

        dense_by_id = {c.id: (c, s) for c, s in dense}
        dense_rank = {cid: i for i, cid in enumerate(dense_ids, start=1)}
        lexical_score = dict(lexical)
        lexical_rank = {cid: i for i, cid in enumerate(lexical_ids, start=1)}

        top_ids = sorted(fused, key=lambda i: fused[i], reverse=True)[:limit]
        missing = [i for i in top_ids if i not in dense_by_id]
        hydrated = await self.store.hydrate(kb_id, missing) if missing else {}

        results: list[ScoredChunk] = []
        for chunk_id in top_ids:
            chunk = (
                dense_by_id[chunk_id][0] if chunk_id in dense_by_id else hydrated.get(chunk_id)
            )
            if chunk is None:
                continue
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    score=fused[chunk_id],
                    dense_score=dense_by_id.get(chunk_id, (None, None))[1],
                    lexical_score=lexical_score.get(chunk_id),
                    dense_rank=dense_rank.get(chunk_id),
                    lexical_rank=lexical_rank.get(chunk_id),
                )
            )
        return results
