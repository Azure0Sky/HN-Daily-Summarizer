import os
import logging
from datetime import date
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List

# Except original title, all fields must be consistent with the output of the summarizer
class NewsSummaryReport(BaseModel):
    original_title: str
    translated_title: str
    core_point: str
    community_views: str


class DailyDigestPayload(BaseModel):
    d: date
    summaries: List[NewsSummaryReport]


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


app = FastAPI(title='HN Agent Data Interface', 
              description='接收来自 GitHub Actions 的每日新闻总结数据，并存入向量数据库')


# Security(verify_api_key) forces the endpoint to require a valid API key in the header for authentication
@app.post('/api/webhook/daily_news', dependencies=[Security(verify_api_key)])
async def receive_daily_news(payload: DailyDigestPayload):
    """Receive the daily news summary from GitHub Actions, process it, and store it in the RAG memory"""
    logging.info(f'Received {len(payload.summaries)} news items for date: {payload.d}')

    # Temporary placeholder for actual database storage logic
    for idx, item in enumerate(payload.summaries):
        print(f'[{idx+1}] Saving to DB: {item.translated_title}')

    # FastAPI would convert the returned dictionary to JSON and send it back as the HTTP response
    return {'status': 'success', 'message': 'Data ingested into RAG memory successfully.'}
