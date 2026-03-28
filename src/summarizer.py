import logging
from openai import OpenAI
from pydantic import BaseModel, Field

MODEL_NAME = "Qwen/Qwen3.5"


class SummaryReport(BaseModel):
    translated_title: str
    core_point: str = Field(
        description='用一至两句话总结文章到底讲了什么技术、产品或事件。要求客观、精炼。'
    )
    community_views: str = Field(
        description='概括评论区的主要共识、争议或有价值的补充视角。如果没有评论或评论无价值，请输出\"暂无有价值评论\"。'
    )


def truncate_text(text, max_chars):
    if not text:
        return ''
    return text[:max_chars] + ('...\n[Content Truncated]' if len(text) > max_chars else '')


def generate_summary(title, content, comments, api_key):
    """Call the LLM to generate a summary based on the title, content, and comments."""
    client = OpenAI(base_url='https://chatapi.starlake.tech/v1', api_key=api_key)

    safe_content = truncate_text(content, 6000)

    joined_comments = "\n---\n".join(comments)
    safe_comments = truncate_text(joined_comments, 3000)

    system_prompt = """
    你是一个资深的科技分析师。你的任务是深度阅读 Hacker News 的文章和评论，并提取核心洞察。
    提取的信息必须使用中文严格映射到提供的结构化模式中。
    
    分析原则：
    1. 剔除废话，保留硬核技术细节或商业逻辑。
    2. 基于给定的文本进行提取，严禁产生幻觉或编造外部信息。
    """

    user_content = f"""
    Title: {title}
    
    Article Content:
    {safe_content}
    
    Hacker News Comments:
    {safe_comments}
    """

    try:
        # response = client.chat.completions.create(
        #     model=MODEL_NAME,
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_content}
        #     ],
        #     temperature=0.2,
        #     max_completion_tokens=3000,
        #     timeout=420
        # )

        response = client.chat.completions.parse(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.2,
            timeout=480,
            response_format=SummaryReport
        )

        parsed_data: SummaryReport = response.choices[0].message.parsed

        if parsed_data is not None:
            # Format the final markdown output for Telegram
            final_md = (
                f"**【{title}】**\n"
                f'**【{parsed_data.translated_title}】**\n'
                f"- 📰 **核心要点**: {parsed_data.core_point}\n"
                f"- 💬 **社区观点**: {parsed_data.community_views}"
            )
            return final_md

        else:
            logging.warning(f'LLM returned no parsable data for "{title}". Response content: {response.choices[0].message.content}')
            return f'**{title}**\n- 📰 **核心要点**: [大模型总结失败]\n- 💬 **社区观点**: [大模型总结失败]'

    except Exception as e:
        logging.error(f'LLM generation failed for "{title}": {e}')
        # If the LLM call fails, return a fallback summary that at least includes the title
        return f'**{title}**\n- 📰 **核心要点**: [大模型总结失败]\n- 💬 **社区观点**: [大模型总结失败]'
