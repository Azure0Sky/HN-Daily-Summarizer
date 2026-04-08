import os
import re
import time
import html
import pickle
import logging
import hashlib
import requests
import trafilatura
from datetime import date
from rank_bm25 import BM25Okapi

import src.rag.utils as rag_utils
from src.infrastructure.database import get_chroma_collection
from src.config.constants import HN_API_BASE, HN_DIGEST_COLLECTION_NAME, BM25_STORE_PATH

REQUEST_TIMEOUT = 10


def fetch_hn_top_stories(limit: int = 10) -> list[dict]:
    """Get top stories from Hacker News. Returns a list of story dicts."""
    try:
        # Get up to 10 top story IDs
        response = requests.get(f'{HN_API_BASE}/topstories.json', timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        story_ids = response.json()[:limit]

        stories = []
        for sid in story_ids:
            # Get each story's metadata (title, url, etc.)
            item_resp = requests.get(f'{HN_API_BASE}/item/{sid}.json', timeout=REQUEST_TIMEOUT)
            if item_resp.status_code == 200:
                stories.append(item_resp.json())
        return stories

    except Exception as e:
        logging.error(f'Failed to fetch top stories: {e}')
        return []


def extract_article_text(url):
    """Extract the main text from an external webpage"""
    if not url:
        return ''
    
    # Filter out resources that are clearly not text-based
    if url.endswith(('.pdf', '.png', '.jpg', '.mp4')) or 'youtube.com/watch' in url:
        return '[Non-text resource or Video]'

    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            # Extract the main text, removing navigation bars, ads, etc.
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            return text if text else '[Failed to extract meaningful text]'
        return '[Fetch failed or blocked by anti-scraping]'

    except Exception as e:
        logging.warning(f'Trafilatura extraction failed for {url}: {e}')
        return '[Extraction Error]'


def _clean_html(raw_html):
    """Clean the simple HTML tags in HN comments"""
    if not raw_html:
        return ''
    # HN comments often contain basic HTML tags like <p>, <a>, <i>, etc.
    # Need to preserve the text but remove the tags.
    text = html.unescape(raw_html)
    text = re.sub(r'<[^>]+>', ' ', text)
    return text.strip()


def _get_top_comments(story_id, kids, limit=10, fetch_kids=True, kids_num=3):
    """
    Use HN API to get the top comments and their replies optionally.
    HN comments are tree-structured. Fetch 'limit' comments and their following replies if fetch_kids is True.
    """
    if not kids:
        return []

    comments = []
    for comment_id in kids[:limit]:
        time.sleep(.5)  # Rate limiting: Avoid hitting API limits, adjust as needed

        try:
            resp = requests.get(f'{HN_API_BASE}/item/{comment_id}.json', timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                comment_data = resp.json()
                # Ensure the node is not deleted and contains text
                if comment_data and not comment_data.get('deleted') and 'text' in comment_data:
                    clean_text = _clean_html(comment_data['text'])
                    # Avoid single comments that are too long and consume too many Tokens
                    comments.append(clean_text[:500] + ('...' if len(clean_text) > 500 else ''))

                    # Optionally fetch one level of replies to the comment
                    if fetch_kids and 'kids' in comment_data:
                        # Only fetch a few replies to control token cost, and do not recursively fetch deeper levels
                        kids_comments = _get_top_comments(story_id, comment_data['kids'], limit=kids_num, fetch_kids=False)
                        comments.extend(kids_comments)

        except Exception as e:
            logging.warning(f'Failed to fetch comment {comment_id}: {e}')
            continue

    logging.info(f'Fetched {len(comments)} comments (including some replies) for story {story_id}')
    return comments


def fetch_story_content(story: dict) -> dict:
    """Assemble the content for a story, including the main text and top comments."""
    url = story.get('url')
    kids = story.get('kids', [])

    content = {
        'text': '',
        'comments': _get_top_comments(story['id'], kids, limit=5)
    }

    if url:
        # 1. external link: Use trafilatura to extract the main text content from the linked webpage
        logging.info(f'{story["id"]}: Extracting external URL: {url}')
        content['text'] = extract_article_text(url)

    else:
        # 2. internal post (Ask HN / Tell HN): Directly read the text field
        logging.info(f'{story["id"]}: Processing internal HN post (Ask HN)')
        raw_text = story.get('text', '')
        if raw_text:
            content['text'] = _clean_html(raw_text)
        else:
            content['text'] = '[No text content provided in this post]'
            
    return content


def _update_bm25_index(new_ids: list, new_documents: list, metadatas: list):
    """Update the BM25 index with new documents. Load the existing index, append new data, and save it back."""
    store = {'ids': [], 'raw_docs': [], 'metadatas': [], 'bm25_obj': None}
    tokenized_corpus = []
    
    # Load historical data if exists
    if os.path.exists(BM25_STORE_PATH):
        with open(BM25_STORE_PATH, 'rb') as f:
            store = pickle.load(f)
            
    # Append new documents
    existed_ids = set(store['ids'])
    for doc_id, doc, meta in zip(new_ids, new_documents, metadatas):
        if doc_id not in existed_ids:
            existed_ids.add(doc_id)
            tokenized_corpus.append(rag_utils.tokenize_for_bm25(doc))

            store['ids'].append(doc_id)
            store['raw_docs'].append(doc)
            store['metadatas'].append(meta)

    # Build BM25 object
    bm25 = BM25Okapi(tokenized_corpus)
    store['bm25_obj'] = bm25  # cache the BM25 object for retrieval use

    tmp_path = f'{BM25_STORE_PATH}.tmp'
    try:
        with open(tmp_path, 'wb') as f:
            pickle.dump(store, f)
        os.replace(tmp_path, BM25_STORE_PATH)  # atomic replace to avoid read/write conflicts

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)  # clean up temp file on failure
        raise RuntimeError(f'Failed to update BM25 index: {e}')


def _save_to_vector_db(ids, documents, metadatas):
    """Save the documents to ChromaDB with the given metadata and IDs."""
    collection = get_chroma_collection(HN_DIGEST_COLLECTION_NAME)
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)


def _delete_from_vector_db(ids):
    """Delete documents from ChromaDB by their IDs."""
    collection = get_chroma_collection(HN_DIGEST_COLLECTION_NAME)
    collection.delete(ids=ids)


def ingest_daily_news(digest_date: date, summaries):
    documents, metadatas, ids = [], [], []

    for news in summaries:
        text_content = (
            f'原标题: {news.original_title}\n'
            f'中译标题: {news.translated_title}\n'
            f'核心要点: {news.core_point}\n'
            f'社区观点: {news.community_views}'
        )
        documents.append(text_content)
        
        metadatas.append({
            'date': digest_date.isoformat(),
            'source': 'HackerNews',
            'title': news.original_title
        })

        # Generate a unique ID for each document based on the date and original title
        unique_str = f'{digest_date}-{news.original_title}'.encode('utf-8')
        doc_id = hashlib.md5(unique_str).hexdigest()
        ids.append(doc_id)

    # Handle dual-write problem
    try:
        _save_to_vector_db(ids, documents, metadatas)
        logging.info(f'Saved {len(documents)} documents to Vector DB for date {digest_date}.')
    except Exception as e:
        logging.error(f'Error occurred while saving to vector DB: {e}')
        raise

    try:
        _update_bm25_index(ids, documents, metadatas)
        logging.info(f'Updated BM25 index with {len(documents)} new documents for date {digest_date}.')
    except Exception as e:
        logging.error(f'Error occurred while updating BM25 index: {e}')

        # Rollback vector DB entries to maintain consistency
        try:
            _delete_from_vector_db(ids)
            logging.info(f'Rolled back {len(ids)} documents from Vector DB due to BM25 index update failure.')
        except Exception as rollback_err:
            logging.critical(
                f'Error occurred while rolling back vector DB entries: {rollback_err}. '
                f'Orphaned IDs in Chroma: {ids}'
            )

        raise RuntimeError(f'Ingestion aborted. Rollback executed. Original error: {e}')

    return len(documents)
