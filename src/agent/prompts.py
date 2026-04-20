import textwrap
import zoneinfo
from datetime import datetime


SUMMARY_SYSTEM_PROMPT = """
你是一个资深的科技分析师。你的任务是深度阅读 Hacker News 的文章和评论，并提取核心洞察。
总结的信息必须使用中文严格映射到提供的结构化json模式中。

分析原则：
1. 剔除废话，保留技术细节或商业逻辑。
2. 基于给定的文本进行提取总结，严禁编造外部信息。
"""


def build_summary_messages(title: str, content: str, comments: str) -> list[dict]:
    user_payload = (
        f'Title: {title}\n'
        f'Content: {content[:8000]}\n'
        f'Comments: {comments[:4000]}'
    )
    
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": user_payload}
    ]


TG_BOT_SYSTEM_PROMPT = """
你是一个严谨的中文助手。
在处理用户问题时，请严格遵守以下决策：
1. 根据用户请求，判断用户意图，选择合适的工具辅助你的回答。
2. 若问题与 Hacker News 社区内容相关，必须优先进行数据库检索，并严格遵循返回结果进行回答；如果检索没有结果，请明确回复：“根据我目前的检索，无法提供准确答案”。
3. 若问题需要外部实时信息、公开网页事实或超出 HN 数据库范围，可调用工具进行联网检索；使用工具返回的摘要证据辅助回答，禁止编造。
4. 若使用任何检索工具，回答末尾需附“参考来源”小节。
5. 使用 Markdown 格式，回答简洁清晰。
""".strip()


def _get_current_time() -> str:
    """Return the current datetime in UTC+8."""
    current_time = datetime.now(zoneinfo.ZoneInfo('Asia/Shanghai'))

    formatted_time = current_time.strftime('%Y-%m-%d %H:%M %A')
    return f"当前系统时间: {formatted_time} (UTC+8)"


def build_agent_messages(messages) -> list[dict]:
    system_prompt = textwrap.dedent(f"""
    [环境信息]
    当前系统时间：{_get_current_time()}

    [核心指令]
    {TG_BOT_SYSTEM_PROMPT}
    """)

    return [
        {'role': 'system', 'content': system_prompt},
        *messages
    ]
