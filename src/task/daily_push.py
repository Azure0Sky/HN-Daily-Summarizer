import logging
import requests
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config.constants import API_KEY_NAME
from src.config.settings import settings
from src.rag.ingestion import fetch_hn_top_stories, fetch_story_content
from src.agent.engine import generate_summary_report
from src.infrastructure.telegram_client import send_telegram_message
from src.interfaces.qq_bot.push import is_qq_push_configured, send_qq_message

MAX_STORY_WORKERS = 3


def _push_to_cloud_server(digest_date: date, summaries: list) -> bool:
    if settings.cloud_server_webhook_url is None:
        logging.critical('Cloud server webhook URL is not configured. Skipping push to Cloud server.')
        return False

    payload = {
        'd': digest_date.isoformat(),
        'summaries': summaries
    }
    headers = {
        'Content-Type': 'application/json',
        API_KEY_NAME: settings.cloud_api_secret
    }

    try:
        response = requests.post(settings.cloud_server_webhook_url, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        logging.info(f'Cloud server push succeeded with status {response.status_code}.')
        return True

    except Exception as e:
        logging.error(f'Failed to push to Cloud server: {e}')
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


def _format_qq_reports(reports: list) -> str:
    plain_text_reports = [report.replace('**', '') for report in reports]
    return '🔥 == Hacker News Daily Digest ==\n\n' + '\n\n---\n\n'.join(plain_text_reports)


def _send_alert(text: str):
    send_telegram_message(text)
    if is_qq_push_configured():
        send_qq_message(text)


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
        gen_error = False  # Track if any story generation failed

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
                    gen_error = True
                    continue

        # 3. Assemble report and dispatch to all configured channels in parallel
        if final_reports:
            if gen_error:
                _send_alert('⚠️ 今日摘要生成过程中部分新闻处理失败。请检查系统日志以获取详细信息。')

            daily_digest = _format_tg_reports(final_reports)
            qq_digest = _format_qq_reports(final_reports)
            qq_configured = is_qq_push_configured()

            with ThreadPoolExecutor(max_workers=3) as executor:
                tg_future = executor.submit(send_telegram_message, daily_digest)
                cloud_future = executor.submit(
                    _push_to_cloud_server,
                    digest_date=date.today(),
                    summaries=structured_summaries,
                )
                qq_future = executor.submit(send_qq_message, qq_digest) if qq_configured else None

                errors = []
                if tg_future.result():
                    logging.info('Successfully pushed daily digest to Telegram.')
                else:
                    errors.append('Failed to push daily digest to Telegram.')

                if qq_future is None:
                    logging.warning('QQ Bot push is not configured. Skipping QQ push.')
                elif qq_future.result():
                    logging.info('Successfully pushed daily digest to QQ.')
                else:
                    errors.append('Failed to push daily digest to QQ.')

                if cloud_future.result():
                    logging.info('Successfully pushed structured digest to Cloud server.')
                else:
                    errors.append('Failed to push structured digest to Cloud server.')

                if errors:
                    raise Exception('\n'.join(errors))

        else:
            logging.warning('No summaries generated today.')
            _send_alert('⚠️ 今日未能生成有效的新闻摘要。请检查系统日志以获取详细信息。')

    except Exception as e:
        logging.critical(f'Critical error in workflow: {e}')
        _send_alert('❌ 今日摘要生成过程中发生错误。请检查系统日志以获取详细信息。')
        raise
