"""每轮发送云端请求前，对完整消息历史进行 PII 掩盖。

运行：python -u source/baseline/run_appworld_full_mask.py [task_id]
记录本地处理与 PII 过滤耗时，排除日志写入/刷新及 HTTP 调用；初始化单列。
按实体类型与原文去重，在本地字典中保存检测记录，每轮仅打印首次发现的实体。
字典只用于记录，不缓存脱敏结果；每轮仍重新检测完整历史。
"""

import json

if __package__:
    from .full_mask import PIIMasker
    from .run_appworld import run
else:
    from full_mask import PIIMasker
    from run_appworld import run


class LoggingPIIMasker(PIIMasker):
    variant = "fullmask"

    def __init__(self, score_threshold=0.5):
        super().__init__(score_threshold=score_threshold)
        self.pii_entities = {}
        self.last_new_entities = []
        self.round_number = 0

    def mask_text(self, text):
        if not text:
            return text
        results = self.analyzer.analyze(
            text=text, language="en", score_threshold=self.score_threshold,
        )
        for result in results:
            original = text[result.start:result.end]
            key = (result.entity_type, original)
            if key not in self.pii_entities:
                entity = {
                    "entity_type": result.entity_type,
                    "text": original,
                    "score": float(result.score),
                    "first_seen_round": self.round_number,
                }
                self.pii_entities[key] = entity
                self.last_new_entities.append(entity)
        return self.anonymizer.anonymize(
            text=text, analyzer_results=results, operators=self.operators,
        ).text

    def mask_messages(self, messages):
        if not isinstance(messages, list):
            raise TypeError("Expected a JSON message array")
        self.round_number += 1
        self.last_new_entities = []
        try:
            return super().mask_messages(messages)
        finally:
            print(f"[PII 新增] 本轮: {len(self.last_new_entities)} 个 | "
                  f"累计唯一实体: {len(self.pii_entities)} 个", flush=True)
            for entity in self.last_new_entities:
                print("  " + json.dumps(entity, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run(masker_factory=LoggingPIIMasker)
