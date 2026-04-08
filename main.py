import sys
import logging
import argparse


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def main():
    parser = argparse.ArgumentParser(description='HackerNews Agent Entrypoint')
    subparsers = parser.add_subparsers(dest='command', required=True)

    subparsers.add_parser('bot', help='Start the Telegram Agent Bot')
    subparsers.add_parser('api', help='Start the FastAPI data receiving endpoint')

    task_parser = subparsers.add_parser('task', help='Run scheduled tasks')
    task_parser.add_argument('task_name', choices=['daily_push'], help='Run the daily HN news fetching and pushing task')

    args = parser.parse_args()
    
    try:
        if args.command == 'api':
            import src.interfaces.api.routes as api_server
            api_server.run_api_server()

        elif args.command == 'bot':
            import src.interfaces.tg_bot.server as tg_bot
            tg_bot.run_tg_bot()

        elif args.command == 'task':
            if args.task_name == 'daily_push':
                import src.task.daily_push as daily_push
                daily_push.run_daily_work()

    except KeyboardInterrupt:
        logging.info('Process interrupted by user.')
        sys.exit(0)

    except Exception as e:
        logging.critical(f'Process crashed: {e}', exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
