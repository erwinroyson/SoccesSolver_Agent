from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env_file():
    """Tiny .env reader (no extra dependency). Real environment variables win."""
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                v = v.strip()
                if v[:1] in "\"'":
                    v = v.strip(v[0])
                else:
                    v = v.split(" #")[0].split("\t#")[0].strip()
                os.environ.setdefault(k.strip(), v)


_load_env_file()

DATA_DIR = Path(os.getenv("DAGENT_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
DB_PATH = Path(os.getenv("DAGENT_DB", DATA_DIR / "state.db"))
OUTBOX = DATA_DIR / "outbox"

# Public data source (every figure the agent shows is tagged with it)
TM_BASE_URL = os.getenv("TM_BASE_URL", "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data")
TM_FILES = ["players", "player_valuations", "appearances"]
TM_SOURCE = "Transfermarkt public dataset (dcaribou/transfermarkt-datasets)"

# Model assumptions (stated in the README)
LEAGUE_ALPHA = 0.5        # how hard production is discounted for weaker leagues
SHRINK_MIN = 600          # minutes of "prior" used to pull small samples towards the average
WINDOW_DAYS = 365
VALUE_MAX_AGE_DAYS = 540  # ignore valuations older than this

# LLM (Gemini via its OpenAI-compatible endpoint, or OpenAI)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
PRICE_IN = float(os.getenv("LLM_PRICE_IN", "0"))    # USD per 1M tokens, for the cost log
PRICE_OUT = float(os.getenv("LLM_PRICE_OUT", "0"))

LEAGUES = {"laliga": "ES1", "la liga": "ES1", "spain": "ES1", "premier league": "GB1", "england": "GB1",
           "bundesliga": "L1", "germany": "L1", "serie a": "IT1", "italy": "IT1", "ligue 1": "FR1",
           "france": "FR1", "eredivisie": "NL1", "netherlands": "NL1", "primeira liga": "PO1", "portugal": "PO1"}
