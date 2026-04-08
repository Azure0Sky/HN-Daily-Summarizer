import logging
import uvicorn
from typing import List
from datetime import date
from pydantic import BaseModel
from fastapi import APIRouter, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader

from src.rag.ingestion import ingest_daily_news
from src.config.settings import settings
from src.config.constants import API_KEY_NAME

router = APIRouter()


# Except original title, all fields must be consistent with the output of the summarizer
class NewsSummaryReport(BaseModel):
    original_title: str
    translated_title: str
    core_point: str
    community_views: str


class DailyDigestPayload(BaseModel):
    d: date
    summaries: List[NewsSummaryReport]


# Github should include the API key in the header of the POST request
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)


def _verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.do_api_secret:
        logging.warning('Unauthorized access attempt to API.')
        raise HTTPException(status_code=403, detail='Could not validate credentials')

    return api_key


@router.post('/webhook/hn_summary', dependencies=[Security(_verify_api_key)])
async def receive_daily_hn_summary(payload: DailyDigestPayload):
    logging.info(f'Processing payload for date: {payload.d}, items number: {len(payload.summaries)}')

    try:
        ingested_count = ingest_daily_news(payload.d, payload.summaries)
        return {'status': 'success', 'ingested_count': ingested_count}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Daily summary ingestion failed: {e}.')


def run_api_server():
    app = FastAPI(title='HN Agent Data Interface')
    app.include_router(router, prefix="/api")

    logging.info(f'Starting FastAPI server on port {settings.fastapi_server_port}...')
    uvicorn.run(app, host='127.0.0.1', port=settings.fastapi_server_port)
