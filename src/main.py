import os
import time
import logging
from fetcher import get_top_stories, fetch_story_content
from summarizer import generate_summary
from notifier import send_telegram_message

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    # 1. Check environment variables
    llm_key = os.getenv('LLM_API_KEY')
    tg_token = os.getenv('TG_BOT_TOKEN')
    tg_chat_id = os.getenv('TG_CHAT_ID')

    if not all([llm_key, tg_token, tg_chat_id]):
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

        # 3. Process each story
        for story in top_stories:
            logging.info(f"\t> Processing story: {story.get('title')}")

            try:
                # Fetch the webpage content and top comments
                content_data = fetch_story_content(story=story)

                # Call the LLM to generate a summary
                # Pass in the title, fetched content, and comments
                summary = generate_summary(
                    title=story.get('title'),
                    content=content_data['text'],
                    comments=content_data['comments'],
                    api_key=llm_key
                )

                final_reports.append(summary)

                time.sleep(2)  # Rate limiting: Avoid hitting API limits, adjust as needed

            except Exception as e:
                logging.error(f"Failed to process story {story.get('id')}: {e}")
                continue

        # 4. Assemble and send the final report to Telegram
        if final_reports:
            daily_digest = '🔥 == Hacker News Daily Digest ==\n\n' + '\n\n---\n\n'.join(final_reports)
            
            send_telegram_message(daily_digest, tg_token, tg_chat_id)
            logging.info('Successfully pushed daily digest to Telegram.')

        else:
            logging.warning('No summaries generated today.')

    except Exception as e:
        logging.critical(f'Critical error in workflow: {e}')
        # 在实际工程中，这里可以向 Telegram 发送一条错误告警，而不是默默失败


if __name__ == "__main__":
    main()
