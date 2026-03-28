import logging
from openai import OpenAI

MODEL_NAME = "Qwen/Qwen3.5"


def truncate_text(text, max_chars):
    if not text:
        return ''
    return text[:max_chars] + ('...\n[Content Truncated]' if len(text) > max_chars else '')


def generate_summary(title, content, comments, api_key):
    """Call the LLM to generate a summary based on the title, content, and comments."""
    client = OpenAI(api_key=api_key)

    safe_content = truncate_text(content, 6000)

    joined_comments = "\n---\n".join(comments)
    safe_comments = truncate_text(joined_comments, 3000)

    system_prompt = """
    你是一个资深的科技媒体编辑。你的任务是根据提供的 Hacker News 帖子标题、正文和评论区内容，输出一份极其精简的结构化中文简报。
    
    输出必须严格遵循以下 Markdown 格式，不要包含任何多余的寒暄语：
    
    **[标题]**
    - 📰 **核心要点**: (用一句话总结文章到底讲了什么技术、产品或事件)
    - 💬 **社区观点**: (概括评论区的主要共识、争议或有价值的补充视角。如果没有评论，请写“暂无有价值评论”)
    """

    user_content = f"""
    Title: {title}
    
    Article Content:
    {safe_content}
    
    Hacker News Comments:
    {safe_comments}
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.2,
            max_tokens=200,
            timeout=15
        )

        summary = response.choices[0].message.content.strip()
        return summary

    except Exception as e:
        logging.error(f'LLM generation failed for "{title}": {e}')
        # If the LLM call fails, return a fallback summary that at least includes the title
        return f'**{title}**\n- 📰 **核心要点**: [大模型总结失败]\n- 💬 **社区观点**: [大模型总结失败]'
