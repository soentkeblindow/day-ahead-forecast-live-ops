import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def get_entsoe_token() -> str:
    token = os.getenv("ENTSOE_API_KEY")
    if not token:
        raise RuntimeError(
            "ENTSOE_API_KEY is not set. Copy .env.example to .env and add your token."
        )
    return token
