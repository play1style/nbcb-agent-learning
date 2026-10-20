"""应用配置模块：从环境变量和 ``.env`` 文件加载运行参数。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _as_bool(value: str | bool | None, default: bool = False) -> bool:
    """将常见环境变量字符串转换为布尔值。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    """模型、存储和 API 服务的运行时配置。"""

    app_name: str = "智能合规助手"
    environment: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-4-flash"
    llm_timeout_seconds: int = 60
    llm_max_retries: int = 2

    embedding_provider: str = "auto"
    zhipu_api_key: str = ""
    zhipu_embedding_url: str = (
        "https://open.bigmodel.cn/api/paas/v4/embeddings"
    )
    embedding_model: str = "embedding-3"
    embedding_timeout_seconds: int = 30
    embedding_batch_size: int = 32
    embedding_dimension: int = 768

    knowledge_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "knowledge")
    vector_store_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "vector_store.json"
    )
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    log_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")

    retrieval_top_k: int = 3
    retrieval_threshold: float = 0.52
    high_confidence_threshold: float = 0.68
    medium_confidence_threshold: float = 0.52
    max_input_length: int = 2000
    memory_window_turns: int = 6

    def __post_init__(self) -> None:
        """将配置中的相对路径统一转换为绝对路径。"""
        # 统一使用绝对路径，避免不同启动目录导致数据写入到错误位置。
        self.knowledge_dir = Path(self.knowledge_dir).expanduser().resolve()
        self.vector_store_path = Path(self.vector_store_path).expanduser().resolve()
        self.data_dir = Path(self.data_dir).expanduser().resolve()
        self.log_dir = Path(self.log_dir).expanduser().resolve()

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "Settings":
        """从 ``.env`` 和进程环境变量创建配置对象。"""
        dotenv_path = Path(env_file) if env_file else PROJECT_ROOT / ".env"
        if dotenv_path.exists():
            try:
                from dotenv import load_dotenv

                # override=False 保证操作系统环境变量优先于 .env 文件。
                load_dotenv(dotenv_path=dotenv_path, override=False)
            except ImportError:
                pass

        def env(name: str, default: str = "") -> str:
            """读取环境变量并返回字符串值。"""
            return os.getenv(name, default)

        return cls(
            app_name=env("APP_NAME", "智能合规助手"),
            environment=env("ENVIRONMENT", "development"),
            host=env("HOST", "0.0.0.0"),
            port=int(env("PORT", "8000")),
            deepseek_api_key=env("DEEPSEEK_API_KEY"),
            deepseek_base_url=env(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
            ),
            deepseek_model=env("DEEPSEEK_MODEL", "deepseek-flash"),
            llm_timeout_seconds=int(env("LLM_TIMEOUT_SECONDS", "60")),
            llm_max_retries=int(env("LLM_MAX_RETRIES", "2")),
            embedding_provider=env("EMBEDDING_PROVIDER", "auto"),
            zhipu_api_key=env("ZHIPU_API_KEY"),
            zhipu_embedding_url=env(
                "ZHIPU_EMBEDDING_URL",
                "https://open.bigmodel.cn/api/paas/v4/embeddings",
            ),
            embedding_model=env("EMBEDDING_MODEL", "embedding-3"),
            embedding_timeout_seconds=int(
                env("EMBEDDING_TIMEOUT_SECONDS", "30")
            ),
            embedding_batch_size=int(env("EMBEDDING_BATCH_SIZE", "32")),
            embedding_dimension=int(env("EMBEDDING_DIMENSION", "768")),
            knowledge_dir=Path(
                env("KNOWLEDGE_DIR", str(PROJECT_ROOT / "knowledge"))
            ),
            vector_store_path=Path(
                env("VECTOR_STORE_PATH", str(PROJECT_ROOT / "vector_store.json"))
            ),
            data_dir=Path(env("DATA_DIR", str(PROJECT_ROOT / "data"))),
            log_dir=Path(env("LOG_DIR", str(PROJECT_ROOT / "logs"))),
            retrieval_top_k=int(env("RETRIEVAL_TOP_K", "3")),
            retrieval_threshold=float(env("RETRIEVAL_THRESHOLD", "0.52")),
            high_confidence_threshold=float(
                env("HIGH_CONFIDENCE_THRESHOLD", "0.68")
            ),
            medium_confidence_threshold=float(
                env("MEDIUM_CONFIDENCE_THRESHOLD", "0.52")
            ),
            max_input_length=int(env("MAX_INPUT_LENGTH", "2000")),
            memory_window_turns=int(env("MEMORY_WINDOW_TURNS", "6")),
        )

    def ensure_directories(self) -> None:
        """创建运行时需要的可写目录。"""
        # 索引、会话、备份和审计文件都要求父目录存在。
        for path in (
            self.knowledge_dir,
            self.vector_store_path.parent,
            self.data_dir,
            self.log_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
