"""应用配置模块——管理 API 密钥、模型参数、知识库路径等运行配置。"""

import os
from dotenv import load_dotenv
from pathlib import Path

# 加载 .env 文件（优先），回退到 .env.example
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=False)
else:
    _example_path = Path(__file__).parent.parent / ".env.example"
    if _example_path.exists():
        load_dotenv(_example_path, override=False)

class Config:
    """合规助手智能体的运行配置。所有核心参数集中在此维护。"""

    # ── DeepSeek API 配置 ──────────────────────────────────────────
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv(
        "DEEPSEEK_API_BASE",
        "https://api.deepseek.com/v1")
    DEEPSEEK_MODEL: str = os.getenv(
        "DEEPSEEK_MODEL",
        "deepseek-v4-flash")

    # ── 模型参数 ────────────────────────────────────────────────────
    MAX_TOKENS = 2048
    TEMPERATURE = 0.1

    # ── 知识库 ──────────────────────────────────────────────────────
    POLICY_FILE: str = os.getenv("KNOWLEDGE_BASE_PATH",str(Path(__file__).parent.parent / "data" / "policies.json"))

    # ── 嵌入模型 ────────────────────────────────────────────────────
    EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    # ── 检索参数 ────────────────────────────────────────────────────
    SEARCH_TOP_K = 3
    SIMILARITY_THRESHOLD = 0.3

    # ── Web 服务 ───────────────────────────────────────────────────
    HOST: str = os.getenv("HOST","0.0.0.0")
    PORT: int = int(os.getenv("PORT",8000))

    # ── 审计日志 ────────────────────────────────────────────────────
    AUDIT_LOG_FILE: str = os.getenv("AUDIT_LOG_PATH",str(Path(__file__).parent.parent / "logs" / "audit.log"))
