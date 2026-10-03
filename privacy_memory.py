"""本地保存完整工具返回；实体位置与替换策略分开存放。"""

import hashlib
import json
import re
from copy import deepcopy


def fingerprint(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


class PrivacyMemory:
    def __init__(self, task):
        self.task = task
        self.records = {}
        self.entity_index = {}
        self._lookup = {}
        self.recent_ids = []

    def touch(self, record_id):
        """按实际工具返回更新最近顺序，重复读取也算一次出现。"""
        if record_id in self.recent_ids:
            self.recent_ids.remove(record_id)
        self.recent_ids.append(record_id)

    def find(self, tool_name, arguments, text):
        key = (fingerprint([tool_name, arguments]), fingerprint(text))
        return self._lookup.get(key)

    def add(self, tool_name, arguments, text, summary, mentions):
        """模型返回实体原值；字符位置由程序定位，不让模型数位置。"""
        record_id = str(len(self.records) + 1)
        spans = set()
        for mention in mentions:
            value, kind = mention['value'], mention['type']
            if not isinstance(value, str) or not value or value not in text:
                raise ValueError(f'抽取结果不是原文中的实体：{value!r}')
            if not isinstance(kind, str) or not kind:
                raise ValueError('实体 type 必须是非空字符串')
            for match in re.finditer(re.escape(value), text):
                spans.add((match.start(), match.end(), kind))

        # 同一位置的嵌套识别只保留最长项，避免重复替换破坏位置。
        accepted = []
        for start, end, kind in sorted(spans, key=lambda s: (-(s[1] - s[0]), s[0], s[2])):
            if not any(start < old_end and old_start < end for old_start, old_end, _ in accepted):
                accepted.append((start, end, kind))
        entities = {
            str(i): {'type': kind, 'start': start, 'end': end}
            for i, (start, end, kind) in enumerate(sorted(accepted))
        }
        call_hash, result_hash = fingerprint([tool_name, arguments]), fingerprint(text)
        self.records[record_id] = {
            'tool_name': tool_name, 'arguments': deepcopy(arguments),
            'call_hash': call_hash, 'result_hash': result_hash,
            'text': text, 'summary': summary, 'entities': entities,
            'policies': {}, 'sanitized_text': None, 'sent_versions': [],
        }
        self._lookup[(call_hash, result_hash)] = record_id
        self.touch(record_id)
        for entity_id, entity in entities.items():
            self.index(record_id, entity_id, text[entity['start']:entity['end']])
        return record_id

    def index(self, record_id, entity_id, value):
        """原值及生成过的代称用于找候选，不代表同名者是同一个人。"""
        if not value:
            return
        kind = self.records[record_id]['entities'][entity_id]['type']
        refs = self.entity_index.setdefault(kind, {}).setdefault(value, [])
        ref = {'record_id': record_id, 'entity_id': entity_id}
        if ref not in refs:
            refs.append(ref)

    def render(self, record_id):
        record = self.records[record_id]
        if record['sanitized_text'] is not None:
            return record['sanitized_text']
        text = record['text']
        for entity_id, entity in sorted(record['entities'].items(), key=lambda x: x[1]['start'], reverse=True):
            policy = record['policies'].get(entity_id)
            if policy is None:
                raise RuntimeError(f'记录 {record_id} 的实体 {entity_id} 尚未确定策略')
            text = text[:entity['start']] + policy['replacement'] + text[entity['end']:]
        record['sanitized_text'] = text
        return text

    def mark_sent(self, texts):
        """由调用方在实际发送后登记；保存传入快照，而非当前策略的版本。"""
        if any(record_id not in self.records for record_id in texts):
            raise ValueError('发送快照包含未知记录')
        if any(not isinstance(text, str) for text in texts.values()):
            raise TypeError('发送快照必须是记录编号到文本的映射')
        for record_id, text in texts.items():
            versions = self.records[record_id]['sent_versions']
            if text not in versions:
                versions.append(text)

    def as_dict(self):
        return {'task': self.task, 'records': self.records, 'entity_index': self.entity_index}
