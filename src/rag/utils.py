import jieba


def truncate_context(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + '\n\n[Context truncated due to length]'


def truncate_compact_text(text: str, limit: int) -> str:
    compact = ' '.join(text.split())  # Delete extra whitespace and newlines
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + '...'


def tokenize_for_bm25(text: str) -> list[str]:
    """
    Use jieba's cut_for_search to tokenize the input text for BM25 retrieval.
    """
    if not text:
        return []

    tokens = jieba.cut_for_search(text.lower())
    return [t.strip() for t in tokens if len(t.strip()) > 0]
