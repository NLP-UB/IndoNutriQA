from typing import List, Dict, Any
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from ...config.config import LOGGER, PII_ENTITIES, INDONESIAN_PII_REPLACEMENTS


class TextAnonymizer:
	"""Wrapper around Presidio for detection and anonymization."""

	def __init__(self, language: str = "id"):
		self.language = language
		self.analyzer = AnalyzerEngine()
		self.anonymizer = AnonymizerEngine()

	def anonymize(self, text: str, entities_to_anonymize: List[str] = None) -> Dict[str, Any]:
		if entities_to_anonymize is None:
			entities_to_anonymize = PII_ENTITIES
		try:
			analysis = self.analyzer.analyze(text=text, language=self.language, entities=entities_to_anonymize)
		except Exception as exc:
			LOGGER.warning("Analyzer failed; returning original text. Error: %s", exc)
			return {"anonymized_text": text, "entities_detected": []}

		operators = {}
		for ent in entities_to_anonymize:
			replacement = INDONESIAN_PII_REPLACEMENTS.get(ent, f"<{ent}>")
			operators[ent] = OperatorConfig("replace", {"new_value": replacement})

		try:
			anonymized = self.anonymizer.anonymize(text=text, analyzer_results=analysis, operators=operators)
		except Exception as exc:
			LOGGER.warning("Anonymizer failed; returning original text. Error: %s", exc)
			return {"anonymized_text": text, "entities_detected": []}

		return {
			"anonymized_text": anonymized.text,
			"entities_detected": [
				{
					"entity_type": r.entity_type,
					"start": r.start,
					"end": r.end,
					"score": r.score,
				}
				for r in analysis
			],
		}
