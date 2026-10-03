"""Stateless English PII masking: every detected entity becomes [REDACTED].

Install: pip install -r source/baseline/requirements.txt
         python -m spacy download en_core_web_lg
Run:     python source/baseline/full_mask.py < request.txt
         python source/baseline/full_mask.py --messages < messages.json

Messages use the OpenAI-style dict format (text only). Original messages stay
local; pass mask_messages(messages) to the cloud on EVERY model call. This is
detector-based masking, not a guarantee of finding all private data. Dates and
locations may be masked even when useful. No relevance analysis or result cache.
"""

import argparse
from copy import deepcopy
from importlib.util import find_spec
import json
import sys


class PIIMasker:
    def __init__(self, score_threshold=0.5):
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import SpacyNlpEngine
        from presidio_anonymizer import AnonymizerEngine
        from presidio_anonymizer.entities import OperatorConfig

        if find_spec("en_core_web_lg") is None:
            raise RuntimeError("Install the local model: python -m spacy download en_core_web_lg")
        nlp = SpacyNlpEngine(models=[{"lang_code": "en", "model_name": "en_core_web_lg"}])
        self.analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
        self.anonymizer = AnonymizerEngine()
        self.operators = {"DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"})}
        self.score_threshold = score_threshold

    def mask_text(self, text):
        if not text:
            return text
        results = self.analyzer.analyze(
            text=text, language="en", score_threshold=self.score_threshold,
        )
        return self.anonymizer.anonymize(
            text=text, analyzer_results=results, operators=self.operators,
        ).text

    def _mask_values(self, value):
        """Keep JSON parameter keys; mask detected entities in data values."""
        if isinstance(value, str):
            return self.mask_text(value)
        if type(value) in (int, float):
            masked = self.mask_text(str(value))
            return value if masked == str(value) else masked
        if isinstance(value, list):
            return [self._mask_values(item) for item in value]
        if isinstance(value, dict):
            return {key: self._mask_values(item) for key, item in value.items()}
        return value

    def mask_messages(self, messages):
        """Rescan the complete message list; preserve roles and tool-call IDs."""
        if not isinstance(messages, list):
            raise TypeError("Expected a JSON message array")
        safe = deepcopy(messages)
        for message in safe:
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = self.mask_text(content)
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "text":
                        raise ValueError("Only text content blocks are supported")
                    block["text"] = self.mask_text(block["text"])
            elif content is not None:
                raise TypeError("Message content must be text, text blocks, or null")
            functions = [call["function"] for call in message.get("tool_calls", [])]
            if "function_call" in message:
                functions.append(message["function_call"])
            for function in functions:
                arguments = json.loads(function["arguments"])
                function["arguments"] = json.dumps(self._mask_values(arguments), ensure_ascii=False)
        return safe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages", action="store_true", help="Read a JSON message array")
    args = parser.parse_args()
    masker = PIIMasker()
    data = sys.stdin.read()
    if args.messages:
        print(json.dumps(masker.mask_messages(json.loads(data)), ensure_ascii=False, indent=2))
    else:
        print(masker.mask_text(data), end="")
