import logging
import requests
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config.constants import API_KEY_NAME
from src.config.settings import settings
from src.rag.ingestion import fetch_hn_top_stories, fetch_story_content
from src.agent.engine import generate_summary_report
from src.infrastructure.telegram_client import send_telegram_message

MAX_STORY_WORKERS = 3


def _push_to_do_server(digest_date: date, summaries: list) -> bool:
    if settings.do_server_webhook_url is None:
        logging.critical('DO server webhook URL is not configured. Skipping push to DO server.')
        return False

    payload = {
        'd': digest_date.isoformat(),
        'summaries': summaries
    }
    headers = {
        'Content-Type': 'application/json',
        API_KEY_NAME: settings.do_api_secret
    }

    try:
        response = requests.post(settings.do_server_webhook_url, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        logging.info(f'DO server push succeeded with status {response.status_code}.')
        return True

    except Exception as e:
        logging.error(f'Failed to push to DO server: {e}')
        return False


def _format_summary_markdown(original_title, summary):
    return (
        f'**【{original_title}】**\n'
        f'**【{summary.translated_title}】**\n'
        f'- 📰 **核心要点**: {summary.core_point}\n'
        f'- 💬 **社区观点**: {summary.community_views}'
    )


def _format_tg_reports(reports: list) -> str:
    return '🔥 == Hacker News Daily Digest ==\n\n' + '\n\n---\n\n'.join(reports)


def _process_story(story: dict):
    story_title = story.get('title', '[Untitled]')
    logging.info(f'\t> Processing story: {story_title}')

    content_data = fetch_story_content(story=story)
    summary = generate_summary_report(
        title=story_title,
        content=content_data['text'],
        comments=content_data['comments'],
    )

    markdown_report = _format_summary_markdown(story_title, summary)
    structured_summary = {
        'original_title': story_title,
        'translated_title': summary.translated_title,
        'core_point': summary.core_point,
        'community_views': summary.community_views
    }
    return markdown_report, structured_summary


def run_daily_work():
    logging.info('Starting HN Daily Summarizer Workflow...')

    try:
        # 1. Fetch data: Only fetch Top 10 to control Token cost and execution time
        top_stories = fetch_hn_top_stories(limit=10)
        if not top_stories:
            logging.warning('No stories fetched. Exiting.')
            return

        final_reports = []
        structured_summaries = []

        # 2. Process stories with small bounded concurrency to reduce total runtime.
        with ThreadPoolExecutor(max_workers=MAX_STORY_WORKERS) as executor:
            future_to_story = {
                executor.submit(_process_story, story): story for story in top_stories
            }

            for future in as_completed(future_to_story):
                story = future_to_story[future]
                try:
                    markdown_report, structured_summary = future.result()
                    final_reports.append(markdown_report)
                    structured_summaries.append(structured_summary)
                except Exception as e:
                    logging.error(f'Failed to process story {story.get("id")}: {e}')
                    continue

        # 3. Assemble report and dispatch to Telegram + DO server in parallel
        if final_reports:
            daily_digest = _format_tg_reports(final_reports)

            with ThreadPoolExecutor(max_workers=2) as executor:
                tg_future = executor.submit(send_telegram_message, daily_digest)
                do_future = executor.submit(_push_to_do_server, digest_date=date.today(), summaries=structured_summaries)

                tg_success = tg_future.result()
                if tg_success:
                    logging.info('Successfully pushed daily digest to Telegram.')
                else:
                    logging.warning('Failed to push daily digest to Telegram.')

                do_success = do_future.result()
                if do_success:
                    logging.info('Successfully pushed structured digest to DO server.')
                else:
                    logging.warning('Structured digest was not pushed to DO server.')

        else:
            logging.warning('No summaries generated today.')
            send_telegram_message('⚠️ 今日未能生成有效的新闻摘要。请检查系统日志以获取详细信息。')

    except Exception as e:
        logging.critical(f'Critical error in workflow: {e}')
        send_telegram_message('❌ 今日摘要生成过程中发生错误。请检查系统日志以获取详细信息。')
