from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    tg_bot_token: str
    tg_chat_id: str
    llm_api_key: str
    do_api_secret: str
    do_server_webhook_url: str | None = None

    chroma_server_port: int = 33000
    fastapi_server_port: int = 33001

    llm_base_url: str = 'https://chatapi.starlake.tech/v1'
    chat_model_name: str = 'Qwen/Qwen3.5'
    embedding_model_name: str = 'Qwen/Qwen3-VL-Embedding-2B'

    transcript_dir: str = './transcripts'
    
    model_config = SettingsConfigDict(
        env_file='.env', 
        env_file_encoding='utf-8',
        extra='ignore',
    )


settings = Settings()  # type: ignore
