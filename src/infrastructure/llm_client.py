from typing import Type, TypeVar
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from src.config.settings import settings

T = TypeVar('T', bound=BaseModel)  # Generic type for Pydantic models


class LLMClient:
    _instance = None

    client : OpenAI | None = None
    model_name: str | None = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMClient, cls).__new__(cls)
            cls._instance.client = OpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url
            )
            cls._instance.model_name = settings.chat_model_name
        return cls._instance

    def create(self, messages: list[dict], temperature: float = 0.2) -> str | None:
        # TODO: Token log, error handling, retry logic, etc.
        if self.client is None:
            raise ValueError('LLM client is not initialized properly.')

        if self.model_name is None:
            raise ValueError('LLM model name is not configured properly.')

        if not messages:
            raise ValueError('Messages list cannot be empty.')

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def parse(self, messages: list[dict], response_format: Type[T], temperature: float = 0.2) -> T | None:
        if self.client is None:
            raise ValueError('LLM client is not initialized properly.')

        if self.model_name is None:
            raise ValueError('LLM model name is not configured properly.')

        if not messages:
            raise ValueError('Messages list cannot be empty.')

        response = self.client.chat.completions.parse(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            timeout=480,
            response_format=response_format
        )
        return response.choices[0].message.parsed

    def parse_response(self, messages: list[dict], response_format: Type[T], temperature: float = 0.2) -> T | None:
        """Responses API variant"""
        if self.client is None:
            raise ValueError('LLM client is not initialized properly.')

        if self.model_name is None:
            raise ValueError('LLM model name is not configured properly.')

        if not messages:
            raise ValueError('Messages list cannot be empty.')
        
        response = self.client.responses.parse(
            model=self.model_name,
            input=messages,
            temperature=temperature,
            timeout=480,
            text_format=response_format
        )
        return response.output_parsed


class AsyncLLMClient:
    _instance = None

    client = None
    model_name= None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AsyncLLMClient, cls).__new__(cls)
            cls._instance.client = AsyncOpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url
            )
            cls._instance.model_name = settings.chat_model_name
        return cls._instance

    async def create(self, 
                     messages: list[dict],
                     tools: list[dict] = [],
                     timeout: int = 180,
                     temperature: float = 0.2):
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice='auto' if tools else 'none',
            temperature=temperature,
            timeout=timeout
        )
        return response.choices[0].message

    async def parse(self, 
                    messages: list[dict], 
                    response_format: Type[T],
                    tools: list[dict] = [],
                    timeout: int = 180,
                    temperature: float = 0.2):
        response = await self.client.chat.completions.parse(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice='auto' if tools else 'none',
            temperature=temperature,
            timeout=timeout,
            response_format=response_format
        )
        return response.choices[0].message


# Expose a singleton instance
llm_client = LLMClient()
async_llm_client = AsyncLLMClient()
