import json
import logging
from registry import registry

from src.rag.retriever import retrieve_web_context

DEFAULT_WEB_SEARCH_TOP_K = 5


@registry.register(
    name='search_web',
    description='Searches the public web and returns evidence blocks.',
    parameters={
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': 'Precise search terms for search engines. Must be concise keywords or phrases, avoid long sentences.',
            },
            'max_results': {
                'type': 'integer',
                'description': 'Max number of web results to return.',
                'default': DEFAULT_WEB_SEARCH_TOP_K,
            },
        },
        'required': ['query'],
        'additionalProperties': False,
    }
)
def search_web(query: str, max_results: int = DEFAULT_WEB_SEARCH_TOP_K) -> str:
    """Tool entrypoint: run DDGS web search and return compact evidence JSON."""
    normalized_query = query.strip()
    if not normalized_query:
        return json.dumps({
            'ok': False,
            'has_relevant': False,
            'error': "Argument 'query' must be a non-empty string.",
        }, ensure_ascii=False)

    try:
        context_text = retrieve_web_context(normalized_query, max_results)
    except Exception as e:
        logging.exception('Failed to run web search: %s', e)
        return json.dumps({
            'ok': False,
            'has_relevant': False,
            'error': f'Web search failed: {e}',
        }, ensure_ascii=False)

    if not context_text:
        result = {
            'ok': True,
            'has_relevant': False,
            'message': 'No relevant web results found.',
        }
    else:
        result = {
            'ok': True,
            'has_relevant': True,
            'context': context_text,
        }

    return json.dumps(result, ensure_ascii=False)
