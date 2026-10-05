"""
config.py
---------
Loads environment variables from .env and initialises the Groq SDK client.
All configuration is centralised here so every other module imports from a
single source of truth.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from groq import Groq

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load .env (walks up the directory tree if needed)
# ---------------------------------------------------------------------------
_env_path: Path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    load_dotenv(dotenv_path=_env_path)
    logger.info("Loaded environment from: %s", _env_path)
else:
    # Fallback: let python-dotenv search for a .env in the working directory
    load_dotenv()
    logger.warning(
        ".env not found at %s – falling back to CWD search.", _env_path
    )

# ---------------------------------------------------------------------------
# Validate required secrets
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "").strip()

if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY is missing or empty. "
        "Add it to your .env file or export it as an environment variable."
    )

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
# NOTE: llama-3.3-70b-versatile requires a Groq plan with LLaMA access.
# The model below is the highest-capacity chat model available on this key.
# Switch back to "llama-3.3-70b-versatile" once the key has access.
GROQ_MODEL: str = "openai/gpt-oss-120b"
GROQ_TEMPERATURE: float = 0.7          # balanced creativity / determinism
GROQ_MAX_TOKENS: int = 512             # enough for concise JSON payloads
GROQ_REQUEST_TIMEOUT: float = 30.0    # seconds

# ---------------------------------------------------------------------------
# Initialise the Groq client (singleton)
# ---------------------------------------------------------------------------
groq_client: Groq = Groq(
    api_key=GROQ_API_KEY,
    timeout=GROQ_REQUEST_TIMEOUT,
)

logger.info(
    "Groq client initialised | model=%s | timeout=%.1fs",
    GROQ_MODEL,
    GROQ_REQUEST_TIMEOUT,
)
