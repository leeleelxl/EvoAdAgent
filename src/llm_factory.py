"""Unified LLM factory — one interface, multiple providers."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.config import LLMConfig


def create_llm(config: LLMConfig) -> ChatOpenAI:
    """Create a ChatOpenAI-compatible LLM from config.

    All providers (OpenAI, Qwen, DeepSeek) expose OpenAI-compatible APIs,
    so we use ChatOpenAI with different base_url.

    Usage:
        llm = create_llm(LLMConfig(provider="openai", model="gpt-4o-mini"))
        llm = create_llm(LLMConfig(provider="qwen", model="qwen-plus"))
        llm = create_llm(LLMConfig(provider="deepseek", model="deepseek-chat"))
    """
    config = config.resolve()
    return ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
