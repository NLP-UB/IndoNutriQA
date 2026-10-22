from datasets import Dataset
from typing import Dict, Any
import json

from ...config.config import LOGGER


class HuggingFaceFormatter:
	"""Convert SQuAD-style JSON to Hugging Face Dataset and push to hub."""

	@staticmethod
	def load_squad_format(json_path: str) -> Dataset:
		with open(json_path, "r", encoding="utf-8") as f:
			payload = json.load(f)
		examples = {"id": [], "context": [], "question": [], "answers": []}
		for article in payload.get("data", []):
			for paragraph in article.get("paragraphs", []):
				context = paragraph.get("context", "")
				for qa in paragraph.get("qas", []):
					examples["id"].append(qa.get("id"))
					examples["context"].append(context)
					examples["question"].append(qa.get("question"))
					examples["answers"].append(qa.get("answers"))
		return Dataset.from_dict(examples)

	@staticmethod
	def push_to_hub(dataset: Dataset, repo_id: str, token: str = None) -> None:
		dataset.push_to_hub(repo_id, token=token)
		LOGGER.info("Dataset pushed to https://huggingface.co/datasets/%s", repo_id)
