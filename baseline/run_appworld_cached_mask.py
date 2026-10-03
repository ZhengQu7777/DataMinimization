"""Presidio baseline: sanitize new messages and reuse earlier sanitized history.

Run: python -u source/baseline/run_appworld_cached_mask.py [task_id]
Repeated tools reuse sanitization only when name, JSON arguments AND raw result
match. Tools still execute normally because AppWorld Python has mutable state.
The cache lasts for one task and never updates an earlier masking decision.
"""

from copy import deepcopy
import json

if __package__:
    from .full_mask import PIIMasker
    from .run_appworld import run
else:
    from full_mask import PIIMasker
    from run_appworld import run


class CachedPIIMasker(PIIMasker):
    variant = "cached_mask"

    def __init__(self, score_threshold=0.5):
        super().__init__(score_threshold=score_threshold)
        self._history = []
        self._safe_history = []
        self._cache = {}
        self.last_counts = {}

    @staticmethod
    def _function_key(function):
        value = deepcopy(function)
        value["arguments"] = json.loads(value["arguments"])
        return value

    def _cache_key(self, message, tools):
        value = deepcopy(message)
        if value.get("role") == "tool":
            call_id = value.pop("tool_call_id", None)
            if call_id not in tools:
                return None  # An unpaired result cannot establish tool identity.
            source = tools[call_id]
        else:
            source = None
        for call in value.get("tool_calls", []):
            call.pop("id", None)
            call["function"] = self._function_key(call["function"])
        if "function_call" in value:
            value["function_call"] = self._function_key(value["function_call"])
        return json.dumps([source, value], sort_keys=True, ensure_ascii=False)

    @staticmethod
    def _with_current_ids(cached, original):
        safe = deepcopy(cached)
        if "tool_call_id" in original:
            safe["tool_call_id"] = original["tool_call_id"]
        for call, original_call in zip(safe.get("tool_calls", []), original.get("tool_calls", [])):
            if "id" in original_call:
                call["id"] = original_call["id"]
            else:
                call.pop("id", None)
        return safe

    def mask_messages(self, messages):
        if not isinstance(messages, list):
            raise TypeError("Expected a JSON message array")

        prefix = 0
        for old, current in zip(self._history, messages):
            if old != current:
                break
            prefix += 1
        if prefix == 0:
            self._cache.clear()
        self._history = self._history[:prefix]
        self._safe_history = self._safe_history[:prefix]
        counts = dict(processed=0, reused_history=prefix, reused_messages=0, reused_tool_results=0)

        tools = {}
        for message in messages[:prefix]:
            for call in message.get("tool_calls", []):
                tools[call["id"]] = self._function_key(call["function"])

        for message in messages[prefix:]:
            key = self._cache_key(message, tools)
            if key is not None and key in self._cache:
                safe = self._with_current_ids(self._cache[key], message)
                counts["reused_messages"] += 1
                counts["reused_tool_results"] += int(message.get("role") == "tool")
            else:
                safe = super().mask_messages([message])[0]
                counts["processed"] += 1
                if key is not None:
                    self._cache[key] = deepcopy(safe)
            self._history.append(deepcopy(message))
            self._safe_history.append(safe)
            for call in message.get("tool_calls", []):
                tools[call["id"]] = self._function_key(call["function"])

        self.last_counts = counts
        return deepcopy(self._safe_history)


if __name__ == "__main__":
    run(masker_factory=CachedPIIMasker)
