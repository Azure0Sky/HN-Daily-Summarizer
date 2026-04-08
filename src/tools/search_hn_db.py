import json
from src.tools.registry import registry
from src.rag.retriever import hybrid_retrieve


@registry.register(
    name='search_hn_db',
    description=(
        'Searches the Hacker News (HN) digest vector database and returns '
        'relevant context chunks with date/title metadata for grounded answers.\n'
        '[Include]: Only recently scraped HN tech news, and community discussions are included.\n'
    ),
    parameters={
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': 'The user question to search in the HN digest memory.',
            },
            # TODO: May add top_k and distance_threshold parameters later for more flexible retrieval control.
        },
        'required': ['query'],
        'additionalProperties': False,
    }
)
def search_hn_database(query: str) -> str:
    """Tool entrypoint: search HN digest vector DB and return serialized retrieval results."""
    query = query.strip()
    if not query:
        return json.dumps({
            'ok': False,
            'has_relevant': False,
            'error': "Argument 'query' must be a non-empty string.",
        }, ensure_ascii=False)

    context_text = hybrid_retrieve(query)
    if not context_text:
        result = {
            'ok': True,
            'has_relevant': False,
            'context': '',
            'message': 'No sufficiently relevant records found in hn_daily_news.',
        }

    else:
        result = {
            'ok': True,
            'has_relevant': True,
            'context': context_text,
        }

    return json.dumps(result, ensure_ascii=False)
