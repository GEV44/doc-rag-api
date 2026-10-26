"""Settings read from environment variables (see .env.example)."""
import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    data_dir: str = "./data"
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 3
    max_upload_mb: int = 10
    api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    emb_base_url: str = ""
    emb_model: str = ""
    emb_api_key: str = ""
    emb_weight: float = 0.5

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_base_url and self.llm_model)

    @property
    def emb_enabled(self) -> bool:
        return bool(self.emb_base_url and self.emb_model)


def load_settings() -> Settings:
    return Settings(
        data_dir=os.getenv("DATA_DIR", "./data"),
        chunk_size=_int("CHUNK_SIZE", 500),
        chunk_overlap=_int("CHUNK_OVERLAP", 100),
        top_k=_int("TOP_K", 3),
        max_upload_mb=_int("MAX_UPLOAD_MB", 10),
        api_key=os.getenv("API_KEY", ""),
        llm_base_url=os.getenv("LLM_BASE_URL", "").rstrip("/"),
        llm_model=os.getenv("LLM_MODEL", ""),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        emb_base_url=os.getenv("EMBEDDINGS_BASE_URL", "").rstrip("/"),
        emb_model=os.getenv("EMBEDDINGS_MODEL", ""),
        emb_api_key=os.getenv("EMBEDDINGS_API_KEY", ""),
        emb_weight=_float("EMBEDDINGS_WEIGHT", 0.5),
    )
