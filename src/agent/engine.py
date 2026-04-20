import time
import json
import logging
from pathlib import Path
from typing import Optional, Any, Dict
from pydantic import BaseModel, Field

from src.config.settings import settings
from src.tools import get_tool_schemas, run_tool_call
from src.infrastructure.llm_client import llm_client, async_llm_client
from src.agent.prompts import build_summary_messages, build_agent_messages


TRANSCRIPT_DIR = None
if settings.transcript_dir:
    TRANSCRIPT_DIR = Path(settings.transcript_dir)
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
else:
    logging.warning('No transcript directory configured. Conversation history will not be saved.')


class SummaryReport(BaseModel):
    translated_title: str
    core_point: str = Field(
        description='用一至两句话总结文章到底讲了什么技术、产品或事件。要求客观、精炼。'
    )
    community_views: str = Field(
        description='概括评论区的主要共识、争议或有价值的补充视角。如果没有评论或评论无价值，请输出\"暂无有价值评论\"。'
    )
    tags: Optional[str] = Field(
        description='请提取2-5个简短标签，标签之间用英文逗号分隔。如果无法提取，请输出[无标签]。'
    )


def generate_summary_report(title: str, content: str, comments: str) -> SummaryReport:
    messages = build_summary_messages(title, content, comments)

    try:
        for _ in range(3):  # Retry up to 3 times if the generated core point seems invalid
            parsed_report = llm_client.parse(
                messages=messages,
                response_format=SummaryReport,
                temperature=0.1
            )

            if len(parsed_report.core_point) < 9:
                logging.warning(f'LLM generated core point is too short, likely invalid. Retrying... Title: "{title}", Report: {parsed_report.model_dump()}')
                messages.append({
                    'role': 'user',
                    'content': '你之前的回答似乎没有正确理解任务要求，生成的要点过于简短。请重新审视输入内容并重新生成更符合要求的格式化总结。'
                })
                continue

            break

        if len(parsed_report.core_point) < 9:
            logging.error(f'LLM failed to generate a valid core point after retries. Title: "{title}", Report: {parsed_report.model_dump()}')

        return parsed_report  # type: ignore

    except Exception as e:
        logging.error(f'LLM generation failed for "{title}": {e}')
        # Return a safe fallback object so the downstream pipeline can continue.
        return SummaryReport(
            translated_title=title,
            core_point='[大模型总结失败]',
            community_views='[大模型总结失败]',
            tags=None
        )


def _micro_compact_history(history: list, keep_recent: int = 1):
    """
    Clean up the conversation history by keeping only the most recent N tool calls 
    and replacing older ones with a placeholder.
    In-place modification of the history list for now.
    """
    tool_results_count = 0

    # Iterate through history in reverse to keep the most recent tool calls intact, and compact older ones.
    for msg in reversed(history):
        if msg.get('role') == 'tool':
            tool_results_count += 1
            if tool_results_count > keep_recent:
                # For older tool calls beyond the most recent N, replace content with a placeholder to save space.
                tool_name = msg.get('tool_name', 'unknown_tool')
                msg['content'] = f'[执行历史工具 {tool_name} 调用，结果已折叠]'


async def auto_compact_history(chat_id: str, history: list):
    # TODO: Apply this strategy when history exceeds certain token count

    if not TRANSCRIPT_DIR:
        logging.warning('Transcript directory is not configured. Skipping auto-compaction of history.')
        return history[-10:]

    chat_transcript_dir = TRANSCRIPT_DIR / chat_id
    chat_transcript_dir.mkdir(exist_ok=True)

    # Write the full history to a timestamped transcript file for this chat session
    chat_transcript_dir.joinpath(f'{int(time.time())}.jsonl').write_text(
        '\n'.join(json.dumps(msg, ensure_ascii=False) for msg in history),
        encoding='utf-8'
    )
    
    # Strip tool call results from older messages
    pure_dialogue = [m for m in history if m['role'] in {'user', 'assistant'} and not m.get('tool_calls')]
    conversation_text = json.dumps(pure_dialogue, ensure_ascii=False)

    summary_prompt = '请用简练的中文总结以下对话的上下文、用户的核心关注点以及已经得出的结论。禁止包含任何客套话。'

    try:
        response = await async_llm_client.create(
            messages=[
                {'role': 'system', 'content': summary_prompt},
                {'role': 'user', 'content': conversation_text}
            ],
            temperature=0.1
        )
        summary = response.choices[0].message.content

        return [
            {'role': 'user', 'content': f'[历史对话已压缩]\n此前对话总结：{summary}'},
            {'role': 'assistant', 'content': '已了解之前的上下文。请继续提问。'}
        ]

    except Exception as e:
        logging.exception(f'Auto compaction failed: {e}')
        # return history[-10:]



def _normalize_tool_arguments(raw_arguments: str) -> Dict[str, Any]:
    """Validate and parse the raw JSON string from tool call by LLM into a dict of arguments."""
    if not raw_arguments:
        return {}

    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        logging.error('Failed to decode tool arguments: %s', raw_arguments)
        raise json.JSONDecodeError('Tool arguments must be a valid JSON string.', raw_arguments, 0)

    if not isinstance(parsed, dict):
        # logging.error('Tool arguments must decode to a JSON object.')
        raise ValueError('Tool arguments must decode to a JSON object.')

    return parsed


async def _agent_loop(messages):
    """Core agentic loop: send messages to LLM, handle tool calls, and yield final answer."""

    # Limit the number of turns to prevent infinite loops of tool calling without resolution.
    MAX_TURNS = 7
    turn_count = 0

    try:
        while turn_count < MAX_TURNS:
            turn_count += 1
            logging.info(f'--- Agentic loop turn {turn_count} ---')

            assistant_msg = await async_llm_client.create(
                messages=messages,
                tools=get_tool_schemas(),
                temperature=0.2,
                timeout=480
            )

            messages.append(assistant_msg.model_dump(exclude_unset=True))

            if not assistant_msg.tool_calls:
                content = assistant_msg.content
                if isinstance(content, str) and content.strip():
                    yield content.strip()
                else:
                    yield '⚠️ 我暂时无法生成稳定回答，请稍后重试。'
                # Agent has finished without calling more tools, end the loop.
                return

            yield f'🔧 正在执行工具调用 (第 {turn_count} 轮)...'

            for tool_call in assistant_msg.tool_calls:
                function = getattr(tool_call, 'function', None)
                tool_name = getattr(function, 'name', '')
                raw_arguments = getattr(function, 'arguments', '{}')
            
                logging.info(f'loop {turn_count} - Processing tool call: {tool_name}')

                try:
                    parsed_arguments = _normalize_tool_arguments(raw_arguments or '{}')
                except Exception as e:
                    logging.warning(f'Failed to parse tool arguments for tool "{tool_name}" with raw args: {raw_arguments}')

                    messages.append({
                        'role': 'tool',
                        'tool_call_id': tool_call.id,
                        'tool_name': tool_name,
                        'content': f'工具调用参数解析或校验失败：{e}。请修正参数格式后重试。',
                    })
                    continue

                tool_output = run_tool_call(tool_name, parsed_arguments)
                messages.append({
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'tool_name': tool_name,
                    'content': str(tool_output),
                })

        yield '⚠️ Agent 工具调用已达最大轮数限制，未能生成最终回答。请尝试简化问题或检查工具调用结果。'

    except Exception as e:
        logging.exception(f'Error during agent loop: {e}')
        yield '⚠️ 处理请求时出现异常，暂时无法完成问答。\n请稍后重试。'


async def chat_with_agent(messages: list[dict]):
    _micro_compact_history(messages)  # in-place
    messages_with_sp = build_agent_messages(messages)  # note: this will create a new list with the system prompt

    async for output in _agent_loop(messages_with_sp):
        yield output

    messages.clear()
    messages.extend(messages_with_sp[1:])  # Keep the system prompt separate from the stored history, and update the original list in-place
