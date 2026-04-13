"""EvoAdAgent global configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMConfig:
    """LLM provider configuration. Supports OpenAI, Qwen, DeepSeek."""

    provider: str = "openai"  # openai | qwen | deepseek
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096

    def resolve(self) -> "LLMConfig":
        """Fill in defaults from environment variables."""
        if self.provider == "openai":
            self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            self.base_url = self.base_url or os.getenv("OPENAI_BASE_URL") or None
        elif self.provider == "qwen":
            self.api_key = self.api_key or os.getenv("DASHSCOPE_API_KEY")
            self.base_url = self.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            self.model = self.model if self.model != "gpt-4o-mini" else "qwen-plus"
        elif self.provider == "deepseek":
            self.api_key = self.api_key or os.getenv("DEEPSEEK_API_KEY")
            self.base_url = self.base_url or "https://api.deepseek.com"
            self.model = self.model if self.model != "gpt-4o-mini" else "deepseek-chat"
        return self


@dataclass
class ProjectConfig:
    """Top-level project configuration."""

    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default=None)
    db_path: Path = field(default=None)
    strategy_dir: Path = field(default=None)

    # LLM configs — separate models for different roles
    executor_llm: LLMConfig = field(default_factory=lambda: LLMConfig(model="gpt-4o-mini"))
    reflector_llm: LLMConfig = field(default_factory=lambda: LLMConfig(model="gpt-4o-mini"))
    distiller_llm: LLMConfig = field(default_factory=lambda: LLMConfig(model="gpt-4o"))
    simulator_llm: LLMConfig = field(default_factory=lambda: LLMConfig(model="gpt-4o-mini"))
    # Embedding model for L2 user profiles and L3 strategy semantic search.
    # Set to None to disable FAISS indexing entirely.
    emb_config: LLMConfig | None = field(
        default_factory=lambda: LLMConfig(provider="qwen", model="text-embedding-v2")
    )

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.db_path is None:
            self.db_path = self.project_root / "evo_ad_agent.db"
        if self.strategy_dir is None:
            self.strategy_dir = self.project_root / "strategies"
        self.strategy_dir.mkdir(parents=True, exist_ok=True)

    def resolve_all(self) -> "ProjectConfig":
        """Resolve all LLM configs from env."""
        self.executor_llm.resolve()
        self.reflector_llm.resolve()
        self.distiller_llm.resolve()
        self.simulator_llm.resolve()
        if self.emb_config is not None:
            self.emb_config.resolve()
        return self
