import logging
from src.rag.ingestion import fetch_hn_top_stories, fetch_story_content
from src.agent.engine import generate_summary_report
from src.infrastructure.telegram_client import send_telegram_message


def _format_summary_markdown(original_title, summary):
    return (
        f'**【{original_title}】**\n'
        f'**【{summary.translated_title}】**\n'
        f'- 📰 **核心要点**: {summary.core_point}\n'
        f'- 💬 **社区观点**: {summary.community_views}'
    )


def _format_tg_reports(reports: list) -> str:
    return '🧪 == AI Flow Test Digest ==\n\n' + '\n\n---\n\n'.join(reports)


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
    return markdown_report


def run_test_work():
    logging.info('Starting AI Flow Test Workflow...')

    try:
        # 1. Fetch data: Only fetch Top 1
        top_stories = fetch_hn_top_stories(limit=1)
        if not top_stories:
            logging.warning('No stories fetched. Exiting.')
            return

        final_reports = []
        for story in top_stories:
            try:
                markdown_report = _process_story(story)
                final_reports.append(markdown_report)
            except Exception as e:
                logging.error(f'Failed to process story {story.get("id")}: {e}')
                continue

        # 2. Assemble report and dispatch to Telegram only
        if final_reports:
            test_digest = _format_tg_reports(final_reports)

            success = send_telegram_message(test_digest)
            if success:
                logging.info('Successfully pushed test digest to Telegram.')
            else:
                raise Exception('Failed to push test digest to Telegram.')
        else:
            logging.warning('No summaries generated today.')
            send_telegram_message('⚠️ [AI Test] 未能生成有效的新闻摘要。请检查系统日志以获取详细信息。')

    except Exception as e:
        logging.critical(f'Critical error in test workflow: {e}')
        send_telegram_message('❌ [AI Test] 摘要生成过程中发生错误。请检查系统日志以获取详细信息。')
