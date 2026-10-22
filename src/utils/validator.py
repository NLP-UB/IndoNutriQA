from typing import Dict, Any, List
from pathlib import Path
import json

from config.config import LOGGER


def ensure_file_exists(path: str) -> None:
	p = Path(path)
	if not p.exists() or not p.is_file():
		raise FileNotFoundError(f"File not found: {path}")


def ensure_dir_exists(path: str) -> None:
	p = Path(path)
	p.parent.mkdir(parents=True, exist_ok=True)


def validate_squad_dataset(payload: Dict[str, Any]) -> List[str]:
	errors: List[str] = []
	if not isinstance(payload, dict):
		errors.append("Top-level JSON must be an object")
		return errors
	if "data" not in payload or not isinstance(payload["data"], list):
		errors.append("Missing 'data' array")
		return errors
	for article in payload["data"]:
		if "paragraphs" not in article or not isinstance(article["paragraphs"], list):
			errors.append("Article missing 'paragraphs' array")
			continue
		for paragraph in article["paragraphs"]:
			if "context" not in paragraph or not isinstance(paragraph["context"], str):
				errors.append("Paragraph missing 'context' string")
				continue
			if "qas" not in paragraph or not isinstance(paragraph["qas"], list):
				errors.append("Paragraph missing 'qas' array")
				continue
			for qa in paragraph["qas"]:
				if "id" not in qa or "question" not in qa or "answers" not in qa:
					errors.append("QA missing required fields 'id', 'question', or 'answers'")
					continue
				answers = qa["answers"]
				if not isinstance(answers, dict) or "text" not in answers or "answer_start" not in answers:
					errors.append("Answers must have 'text' and 'answer_start'")
					continue
				if len(answers.get("text", [])) != len(answers.get("answer_start", [])):
					errors.append("Answers 'text' and 'answer_start' length mismatch")
	return errors


def safe_load_json(path: str) -> Any:
	ensure_file_exists(path)
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def safe_write_json(obj: Any, path: str) -> None:
	ensure_dir_exists(path)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(obj, f, ensure_ascii=False, indent=2)
	LOGGER.info("Saved JSON to %s", path)
