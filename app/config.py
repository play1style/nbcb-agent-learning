"""应用配置模块.

集中管理系统配置项，包括API密钥、模型参数、路径等。
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件（优先），回退到 .env.example
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=False)
else:
    _example_path = Path(__file__).parent.parent / ".env.example"
    if _example_path.exists():
        load_dotenv(_example_path, override=False)

class Config:
    """应用配置类.

    读取环境变量或使用默认值初始化配置项。
    """

    # Deepseek API配置
    # API密钥通过 .env 文件配置，不提交到代码仓库
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_API_BASE: str = os.getenv(
        "DEEPSEEK_API_BASE",
        "https://api.deepseek.com"
    )
    DEEPSEEK_MODEL: str = os.getenv(
        "DEEPSEEK_MODEL",
        "deepseek-v4-flash"
    )

    # 知识库配置
    KNOWLEDGE_BASE_PATH: str = os.getenv(
        "KNOWLEDGE_BASE_PATH",
        str(Path(__file__).parent.parent / "data" / "knowledge_base.json")
    )

    # 检索配置
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "3"))
    RETRIEVAL_SIMILARITY_THRESHOLD: float = float(
        os.getenv("RETRIEVAL_SIMILARITY_THRESHOLD", "0.1")
    )

    # 服务器配置
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # 审计日志配置
    AUDIT_LOG_PATH: str = os.getenv(
        "AUDIT_LOG_PATH",
        str(Path(__file__).parent.parent / "logs" / "audit.log")
    )

config = Config()
"""全局配置单例."""
