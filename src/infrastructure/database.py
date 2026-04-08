import chromadb
from typing import Any, cast
from functools import lru_cache
from chromadb.utils import embedding_functions

from src.config.settings import settings


@lru_cache(maxsize=1)
def get_chroma_collection(collection_name: str):
    chroma_client = chromadb.HttpClient(host='127.0.0.1', port=settings.chroma_server_port)

    openai_ef = cast(Any, embedding_functions.OpenAIEmbeddingFunction(
        api_key=settings.llm_api_key,
        model_name=settings.embedding_model_name,
        api_base=settings.llm_base_url
    ))

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=openai_ef
    )
    
    return collection
