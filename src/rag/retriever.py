from ddgs import DDGS
from typing import List, Dict

from src.infrastructure.database import get_chroma_collection
from src.config.constants import HN_DIGEST_COLLECTION_NAME

DEFAULT_DISTANCE_THRESHOLD = 1.5
MAX_CONTEXT_CHARS = 2000

# Web search relevant
MAX_WEB_ITEM_TITLE_CHARS = 50
MAX_WEB_ITEM_SNIPPET_CHARS = 500
MAX_WEB_CONTEXT_CHARS = 3000


def _truncate_context(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + '\n\n[Context truncated due to length]'


def _truncate_compact_text(text: str, limit: int) -> str:
    compact = ' '.join(text.split())  # Delete extra whitespace and newlines
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + '...'


def retrieve_relevant_context(query: str, top_k: int = 5):
    """Retrieve relevant context from Chroma and apply distance-threshold filtering."""
    collection = get_chroma_collection(HN_DIGEST_COLLECTION_NAME)

    query_result = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=['documents', 'metadatas', 'distances'],
    )

    documents = (query_result.get('documents') or [[]])[0] or []
    metadatas = (query_result.get('metadatas') or [[]])[0] or []
    distances = (query_result.get('distances') or [[]])[0] or []

    if not documents:
        return ''

    filtered_chunks = []
    for idx, raw_doc in enumerate(documents):
        if not isinstance(raw_doc, str) or not raw_doc.strip():
            continue

        distance = distances[idx] if idx < len(distances) else None
        metadata = metadatas[idx] if idx < len(metadatas) and isinstance(metadatas[idx], dict) else {}

        # Skip weak matches that exceed the distance threshold
        if isinstance(distance, (int, float)) and distance > DEFAULT_DISTANCE_THRESHOLD:
            continue

        date_value = str(metadata.get('date', 'unknown_date'))
        # title_value = str(metadata.get('title', 'unknown_title'))
        distance_value = f'{float(distance):.4f}' if isinstance(distance, (int, float)) else 'unknown_distance'

        filtered_chunks.append(
            f'chunk {idx + 1} | date: {date_value} | distance: {distance_value}\n{raw_doc.strip()}'
        )

    if not filtered_chunks:
        return ''

    merged_context = '\n\n'.join(filtered_chunks)
    return _truncate_context(merged_context, MAX_CONTEXT_CHARS)


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

        title = _truncate_compact_text(raw_title or 'untitled', MAX_WEB_ITEM_TITLE_CHARS)
        snippet = _truncate_compact_text(raw_snippet or 'No snippet available.', MAX_WEB_ITEM_SNIPPET_CHARS)
        url = raw_url or 'unknown_url'

        condensed_chunks.append(
            f'web_chunk {idx} | title: {title}\nsnippet: {snippet}\nurl: {url}'
        )

        if raw_url:
            source_refs.append({'title': title, 'url': raw_url})

    if not condensed_chunks:
        return ''

    merged_context = '\n\n'.join(condensed_chunks)
    return _truncate_context(merged_context, MAX_WEB_CONTEXT_CHARS)
