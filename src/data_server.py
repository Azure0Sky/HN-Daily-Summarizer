import os
import logging
import hashlib
import chromadb
from typing import List
from datetime import date
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


# Except original title, all fields must be consistent with the output of the summarizer
class NewsSummaryReport(BaseModel):
    original_title: str
    translated_title: str
    core_point: str
    community_views: str


class DailyDigestPayload(BaseModel):
    d: date
    summaries: List[NewsSummaryReport]


# Initialize ChromaDB. Persist data to local directory.
CHROMA_DATA_DIR = './chroma_data'
chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_DIR)

# Configure the OpenAI embedding function
openai_ef = OpenAIEmbeddingFunction(
    api_key=os.getenv('LLM_API_KEY'),
    model_name='Qwen/Qwen3-VL-Embedding-2B'
)

collection = chroma_client.get_or_create_collection(
    name='hn_daily_news',
    embedding_function=openai_ef
)


# Github should include the API key in the header of the POST request, with the key name defined in API_KEY_NAME
API_KEY_NAME = 'X-Action-Secret'
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)


def verify_api_key(api_key: str = Security(api_key_header)):
    # Configured in .env.prod
    expected_key = os.getenv('DO_API_SECRET', 'change_this_to_a_complex_random_string')
    if api_key != expected_key:
        logging.warning('Unauthorized access attempt to API.')
        raise HTTPException(status_code=403, detail='Could not validate credentials')

    return api_key


app = FastAPI(title='HN Agent Data Interface')


@app.post('/api/webhook/daily_news', dependencies=[Security(verify_api_key)])
async def receive_daily_news(payload: DailyDigestPayload):
    logging.info(f'Processing payload for date: {payload.d}, items: {len(payload.summaries)}')

    documents = []
    metadatas = []
    ids = []

    for news_summary in payload.summaries:
        # Construct the text content for embedding
        text_content = f"""
        原标题: {news_summary.original_title}
        中译标题: {news_summary.translated_title}
        核心要点: {news_summary.core_point}
        社区观点: {news_summary.community_views}
        """
        documents.append(text_content)

        metadatas.append({
            'date': payload.d.isoformat(),
            'source': 'HackerNews',
            'title': news_summary.original_title
        })

        # Generate a unique ID for each document based on the date and original title
        unique_str = f'{payload.d}-{news_summary.original_title}'.encode('utf-8')
        doc_id = hashlib.md5(unique_str).hexdigest()
        ids.append(doc_id)

    try:
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        logging.info('Successfully ingested into ChromaDB.')
        return {'status': 'success', 'inserted_count': len(documents)}

    except Exception as e:
        logging.error(f'Failed to insert into ChromaDB: {e}')
        raise HTTPException(status_code=500, detail='Database insertion failed.')
