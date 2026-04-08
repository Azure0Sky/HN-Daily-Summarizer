import os
import pickle
from ddgs import DDGS
from typing import List, Dict

import src.rag.utils as rag_utils
from src.infrastructure.database import get_chroma_collection
from src.config.constants import HN_DIGEST_COLLECTION_NAME, BM25_STORE_PATH

DEFAULT_DISTANCE_THRESHOLD = 1.5
MAX_CONTEXT_CHARS = 2000

# Web search relevant
MAX_WEB_ITEM_TITLE_CHARS = 50
MAX_WEB_ITEM_SNIPPET_CHARS = 500
MAX_WEB_CONTEXT_CHARS = 3000


def _get_bm25_store():
    if not os.path.exists(BM25_STORE_PATH):
        return None
    with open(BM25_STORE_PATH, 'rb') as f:
        return pickle.load(f)


def retrieve_web_context(query: str, max_results: int) -> str:
    """Retrieve concise web evidence blocks from DDGS."""
    search_results = list(DDGS().text(query, max_results=max_results))
    if not search_results:
        return ''

    condensed_chunks: List[str] = []
    source_refs: List[Dict[str, str]] = []

    for idx, item in enumerate(search_results, start=1):
        if not isinstance(item, dict):
            continue

        raw_title = str(item.get('title', '')).strip()
        raw_snippet = str(item.get('body', '')).strip()
        raw_url = str(item.get('href', '')).strip()

        if not raw_title and not raw_snippet and not raw_url:
            continue

        title = rag_utils.truncate_compact_text(raw_title or 'untitled', MAX_WEB_ITEM_TITLE_CHARS)
        snippet = rag_utils.truncate_compact_text(raw_snippet or 'No snippet available.', MAX_WEB_ITEM_SNIPPET_CHARS)
        url = raw_url or 'unknown_url'

        condensed_chunks.append(
            f'web_chunk {idx} | title: {title}\nsnippet: {snippet}\nurl: {url}'
        )

        if raw_url:
            source_refs.append({'title': title, 'url': raw_url})

    if not condensed_chunks:
        return ''

    merged_context = '\n\n'.join(condensed_chunks)
    return rag_utils.truncate_context(merged_context, MAX_WEB_CONTEXT_CHARS)


def _dense_retrieve(query: str, top_k: int):
    collection = get_chroma_collection(HN_DIGEST_COLLECTION_NAME)

    query_result = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=['documents', 'metadatas', 'distances'],
    )

    ids = (query_result.get('ids') or [[]])[0]
    documents = (query_result.get('documents') or [[]])[0]
    metadatas = (query_result.get('metadatas') or [[]])[0]
    distances = (query_result.get('distances') or [[]])[0]

    if not documents:
        return []

    dense_results = []
    for idx, raw_doc in enumerate(documents):
        if not isinstance(raw_doc, str) or not raw_doc.strip():
            continue

        doc_id = ids[idx] if idx < len(ids) else None
        distance = distances[idx] if idx < len(distances) else None
        metadata = metadatas[idx] if idx < len(metadatas) and isinstance(metadatas[idx], dict) else {}

        # Skip weak matches that exceed the distance threshold
        if isinstance(distance, (int, float)) and distance > DEFAULT_DISTANCE_THRESHOLD:
            continue

        dense_results.append((doc_id, raw_doc.strip(), metadata, distance))

    return dense_results


def _sparse_retrieve(query: str, top_k: int):
    store = _get_bm25_store()
    if store is None or 'bm25_obj' not in store:
        return []

    bm25 = store['bm25_obj']
    tokenized_query = rag_utils.tokenize_for_bm25(query)
    scores = bm25.get_scores(tokenized_query)

    sparse_results = [(store['ids'][i], store['raw_docs'][i], store['metadatas'][i], scores[i])  # [(doc_id, raw_doc, metadata, score), ...]
                  for i in range(len(scores)) if scores[i] > 0]

    # Sort by BM25 score in descending order
    sparse_results.sort(key=lambda x: x[3], reverse=True)

    return sparse_results[:top_k]


def hybrid_retrieve(query: str, top_k: int = 5) -> str:
    # Dense Retrieval from Chroma
    dense_hits = _dense_retrieve(query, top_k * 2)  # Retrieve more for better fusion

    # Sparse Retrieval using BM25
    sparse_hits = _sparse_retrieve(query, top_k * 2)

    # Reciprocal Rank Fusion
    fused_scores = {}
    doc_lookup = {}
    K = 60  # RRF constant

    for rank, (doc_id, doc, meta, *_) in enumerate(dense_hits, start=1):
        if doc_id not in fused_scores:
            fused_scores[doc_id] = 0
            doc_lookup[doc_id] = doc, meta
        fused_scores[doc_id] += 1 / (K + rank)
        
    for rank, (doc_id, doc, meta, *_) in enumerate(sparse_hits, start=1):
        if doc_id not in fused_scores:
            fused_scores[doc_id] = 0
            doc_lookup[doc_id] = doc, meta
        fused_scores[doc_id] += 1 / (K + rank)

    # Sort documents by fused score in descending order
    sorted_fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    chunks = []
    for rank, (d_id, score) in enumerate(sorted_fused[:top_k], start=1):
        doc, metadata = doc_lookup[d_id]

        date_value = str(metadata.get('date', 'unknown_date'))

        chunk_text = (
            f'rank {rank} | date: {date_value} | RRF_score: {score:.4f}\n'
            f'{doc.strip()}'
        )
        chunks.append(chunk_text)

    if not chunks:
        return ''

    merged_context = '\n\n'.join(chunks)
    return rag_utils.truncate_context(merged_context, MAX_WEB_CONTEXT_CHARS)
