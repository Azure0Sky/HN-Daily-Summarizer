import os
import time
import logging
from datetime import date
from concurrent.futures import ThreadPoolExecutor

from fetcher import get_top_stories, fetch_story_content
from summarizer import generate_summary
from notifier import send_telegram_message
from push_service import push_to_do_server

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def format_summary_markdown(original_title, summary):
    return (
        f"**【{original_title}】**\n"
        f"**【{summary.translated_title}】**\n"
        f"- 📰 **核心要点**: {summary.core_point}\n"
        f"- 💬 **社区观点**: {summary.community_views}"
    )


# Run in Github Actions
def main():
    # 1. Check environment variables
    llm_key = os.getenv('LLM_API_KEY')
    # llm_key = os.getenv('OR_LLM_API_KEY')

    tg_token = os.getenv('TG_BOT_TOKEN')
    tg_chat_id = os.getenv('TG_CHAT_ID')
    do_webhook_url = os.getenv('DO_SERVER_WEBHOOK_URL')
    do_api_secret = os.getenv('DO_API_SECRET')

    if not all([llm_key, tg_token, tg_chat_id, do_api_secret]):
        logging.error('Missing critical environment variables. Check GitHub Secrets.')
        return

    logging.info('Starting HN Daily Summarizer Workflow...')

    try:
        # 2. Fetch data: Only fetch Top 10 to control Token cost and execution time
        top_stories = get_top_stories(limit=10)
        if not top_stories:
            logging.warning('No stories fetched. Exiting.')
            return

        final_reports = []
        structured_summaries = []

        # 3. Process each story
        for story in top_stories:
            story_title = story.get('title', '[Untitled]')
            logging.info(f"\t> Processing story: {story_title}")

            try:
                # Fetch the webpage content and top comments
                content_data = fetch_story_content(story=story)

                # Call the LLM to generate a summary
                # Pass in the title, fetched content, and comments
                summary = generate_summary(
                    title=story_title,
                    content=content_data['text'],
                    comments=content_data['comments'],
                    api_key=llm_key
                )

                # 3.1 Keep markdown output for Telegram.
                final_reports.append(format_summary_markdown(story_title, summary))

                # 3.2 Keep structured output for FastAPI ingestion. 
                # Must align with NewsSummaryReport in src/recv_service.py
                structured_summaries.append({
                    'original_title': story_title,
                    'translated_title': summary.translated_title,
                    'core_point': summary.core_point,
                    'community_views': summary.community_views
                })

                time.sleep(1)  # Rate limiting: Avoid hitting API limits, adjust as needed

            except Exception as e:
                logging.error(f"Failed to process story {story.get('id')}: {e}")
                continue

        # 4. Assemble report and dispatch to Telegram + DO server in parallel
        if final_reports:
            daily_digest = '🔥 == Hacker News Daily Digest ==\n\n' + '\n\n---\n\n'.join(final_reports)

            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both outbound tasks together so they can run concurrently.
                tg_future = executor.submit(send_telegram_message, daily_digest, tg_token, tg_chat_id)
                do_future = executor.submit(
                    push_to_do_server,
                    digest_date=date.today(),
                    summaries=structured_summaries,
                    webhook_url=do_webhook_url,
                    api_secret=do_api_secret
                )

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

    except Exception as e:
        logging.critical(f'Critical error in workflow: {e}')  # TODO: May send message to Telegram if this happens


if __name__ == "__main__":
    main()
