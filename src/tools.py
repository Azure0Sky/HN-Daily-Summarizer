import json
import logging
import zoneinfo
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from database import get_chroma_collection


DEFAULT_RETRIEVAL_TOP_K = 5
DEFAULT_DISTANCE_THRESHOLD = 1.0
MAX_CONTEXT_CHARS = 2000


SEARCH_HN_DATABASE_TOOL: Dict[str, Any] = {
    'type': 'function',
    'function': {
        'name': 'search_hn_database',
        'description': (
            'Searches the Hacker News (HN) digest vector database and returns '
            'relevant context chunks with date/title metadata for grounded answers.\n'
            '[Include]: Only recently scraped HN tech news, and community discussions are included.\n'
            '[Not Include]: Not include politics, entertainment, or life news, etc.'
        ),
        'parameters': {
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
        },
    },
}


GET_CURRENT_TIME_TOOL: Dict[str, Any] = {
    'type': 'function',
    'function': {
        'name': 'get_current_time',
        'description': (
            'Returns the current datetime anchor in UTC+8, '
            'for converting relative-time expressions to absolute dates.'
            '[Important]: For relative-time requests (e.g., yesterday / recent two days), '
            'must call this first and convert them to absolute dates.\n'
        ),
        'parameters': {
            'type': 'object',
            'properties': {},
            'additionalProperties': False,
        },
    },
}


# Tool route handlers map: tool name to execution function
TOOL_HANDLERS = {
    'get_current_time': lambda **args: get_current_time(),
    'search_hn_database': lambda **args: search_hn_database(args.get('query', '')),
}


def _truncate_context(text: str) -> str:
    if len(text) <= MAX_CONTEXT_CHARS:
        return text
    return text[:MAX_CONTEXT_CHARS] + '\n\n[Context truncated due to length]'


def _retrieve_relevant_context(user_query: str) -> Tuple[str, bool]:
    """Retrieve relevant context from Chroma and apply distance-threshold filtering."""
    collection = get_chroma_collection()

    query_result = collection.query(
        query_texts=[user_query],
        n_results=DEFAULT_RETRIEVAL_TOP_K,
        include=['documents', 'metadatas', 'distances'],
    )

    documents = (query_result.get('documents') or [[]])[0] or []
    metadatas = (query_result.get('metadatas') or [[]])[0] or []
    distances = (query_result.get('distances') or [[]])[0] or []

    if not documents:
        return '', False

    filtered_chunks: List[str] = []
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
        return '', False

    merged_context = '\n\n'.join(filtered_chunks)
    return _truncate_context(merged_context), True


def get_current_time() -> str:
    """Return the current datetime anchor in UTC+8 for converting relative-time expressions to absolute dates."""
    current_time = datetime.now(zoneinfo.ZoneInfo('Asia/Shanghai'))

    weekdays = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    weekday_str = weekdays[current_time.weekday()]
    
    formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
    return f"当前系统标准时间: {formatted_time} ({weekday_str})"


def search_hn_database(query: str) -> str:
    """Tool entrypoint: search HN digest vector DB and return serialized retrieval results."""
    normalized_query = query.strip()
    if not normalized_query:
        return json.dumps({
            'ok': False,
            'has_relevant': False,
            'error': "Argument 'query' must be a non-empty string.",
        }, ensure_ascii=False)

    context_text, has_relevant = _retrieve_relevant_context(normalized_query)
    if not has_relevant:
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


@lru_cache(maxsize=1)
def get_tool_schemas() -> List[Dict[str, Any]]:
    return [GET_CURRENT_TIME_TOOL, SEARCH_HN_DATABASE_TOOL]


def run_tool_call(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Dispatch and execute tool call by name, then return result text for tool messages."""
    if tool_name not in TOOL_HANDLERS:
        error_msg = f"Tool '{tool_name}' is not recognized. Available tools: {list(TOOL_HANDLERS.keys())}."
        logging.error(error_msg)
        return error_msg

    try:
        result = TOOL_HANDLERS[tool_name](**arguments)
    except Exception as e:
        result = f"Error occurred while running tool '{tool_name}': {e}"

    return result
