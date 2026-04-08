import re
import time
import html
import logging
import hashlib
import requests
import trafilatura
from datetime import date

from src.infrastructure.database import get_chroma_collection
from src.config.constants import HN_API_BASE, HN_DIGEST_COLLECTION_NAME

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
        logging.error(f"Failed to fetch top stories: {e}")
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
    Get the top comments and their replies optionally.
    HN comments are tree-structured. Fetch 'limit' comments and their following replies if fetch_kids is True.
    """
    if not kids:
        return []

    comments = []
    for comment_id in kids[:limit]:
        time.sleep(.5)  # Rate limiting: Avoid hitting API limits, adjust as needed

        try:
            resp = requests.get(f"{HN_API_BASE}/item/{comment_id}.json", timeout=REQUEST_TIMEOUT)
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
            logging.warning(f"Failed to fetch comment {comment_id}: {e}")
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


def ingest_to_vector_db(digest_date: date, summaries):
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

    collection = get_chroma_collection(HN_DIGEST_COLLECTION_NAME)
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    
    return len(documents)
