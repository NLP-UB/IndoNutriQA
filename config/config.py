import os
from pathlib import Path
import logging

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"

# for dir_path in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR):
# 	try:
# 		dir_path.mkdir(parents=True, exist_ok=True)
# 	except Exception as exc:
# 		raise RuntimeError(f"Failed to create directory: {dir_path}") from exc

PII_ENTITIES = [
	"PERSON",
	"PHONE_NUMBER",
	"EMAIL_ADDRESS",
	"LOCATION",
	"DATE_TIME",
	"ID_NUMBER",
]

INDONESIAN_PII_REPLACEMENTS = {
	"PERSON": "<NAMA>",
	"PHONE_NUMBER": "<TELEPON>",
	"EMAIL_ADDRESS": "<EMAIL>",
	"LOCATION": "<LOKASI>",
	"ID_NUMBER": "<ID>",
}

DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Logging setup
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
	level=LOG_LEVEL,
	format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("indonesian_clinical_qa")
