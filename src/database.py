import os
import chromadb
from functools import lru_cache
from typing import Any, cast
from chromadb.utils import embedding_functions


@lru_cache(maxsize=1)
def get_chroma_collection():
    chroma_port = int(os.getenv('CHROMA_SERVER_PORT', 80010))
    # Connect to local ChromaDB server
    # port should match the one configured in systemd service file for the ChromaDB service
    chroma_client = chromadb.HttpClient(host='127.0.0.1', port=chroma_port)

    openai_ef = cast(Any, embedding_functions.OpenAIEmbeddingFunction(
        api_key_env_var='LLM_API_KEY',
        model_name='Qwen/Qwen3-VL-Embedding-2B',
        api_base=os.getenv('LLM_BASE_URL', 'https://chatapi.starlake.tech/v1')
    ))  # cast to Any to bypass type issues with Chroma's embedding function

    collection = chroma_client.get_or_create_collection(
        name='hn_daily_news',
        embedding_function=openai_ef
    )
    
    return collection
