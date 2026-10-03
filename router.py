"""用程序选材料、拼小模型输入；这里不调用模型。"""

import json
import re
from copy import deepcopy


def words(text):
    english = re.findall(r'[a-z0-9_]+', text.lower())
    chinese = re.findall(r'[\u4e00-\u9fff]+', text)
    return set(english + [s[i:i + 2] for s in chinese for i in range(len(s) - 1)])


class Router:
    def __init__(self, history_limit=4):
        self.history_limit = history_limit
        self._terms = {}

    def select(self, memory, cloud_response, current_ids, extra_query=''):
        query = json.dumps(cloud_response, ensure_ascii=False) + extra_query
        query += '\n'.join(memory.records[r]['text'] for r in set(current_ids))
        query_words = words(query)
        exact = set()
        for values in memory.entity_index.values():
            for value, refs in values.items():
                if value in query:
                    exact.update(ref['record_id'] for ref in refs)
        for record_id, record in memory.records.items():
            if record_id not in self._terms:
                self._terms[record_id] = words(json.dumps(
                    [record['tool_name'], record['arguments'], record['text']], ensure_ascii=False))

        old_ids = [r for r in memory.records if r not in current_ids]
        scores = {r: len(query_words & self._terms[r]) for r in old_ids}
        recent = [r for r in memory.recent_ids if r in old_ids][-2:]
        lexical = sorted((r for r in old_ids if scores[r]), key=lambda r: scores[r], reverse=True)[:3]
        candidates = set(recent + lexical) | (exact & set(old_ids))
        ranked = sorted(candidates, key=lambda r: (r in exact, r in recent, scores[r], int(r)), reverse=True)
        return list(dict.fromkeys(current_ids + ranked[:self.history_limit]))

    def build_input(self, memory, cloud_response, selected_ids, new_ids, full_text=False):
        payload = {'task': memory.task, 'cloud_response': deepcopy(cloud_response),
                   'new_tool_results': {}, 'selected_history': {}}
        for record_id in selected_ids:
            record = memory.records[record_id]
            # 正常只读摘要与实体周围原文；需要补读时才提供所选记录全文。
            ranges = [(0, len(record['text']))] if full_text else [
                (max(0, e['start'] - 60), min(len(record['text']), e['end'] + 60))
                for e in record['entities'].values()
            ]
            if not ranges:
                ranges = [(0, min(240, len(record['text'])))]
            merged = []
            for start, end in sorted(ranges):
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            entities = {}
            for entity_id, entity in record['entities'].items():
                policy = record['policies'].get(entity_id)
                entities[entity_id] = {
                    'type': entity['type'],
                    'value': record['text'][entity['start']:entity['end']],
                    'current_policy': ({k: policy[k] for k in ['action', 'replacement']} if policy else None),
                }
            item = {
                'tool_name': record['tool_name'], 'arguments': deepcopy(record['arguments']),
                'summary': record['summary'],
                'snippets': [{'start': s, 'text': record['text'][s:e]} for s, e in merged],
                'entities': entities,
                # 只取已发送版本中对应实体/代称周围的片段，完整快照仍留在本地。
                'sent_snippets': self.sent_snippets(memory, record_id),
            }
            section = 'new_tool_results' if record_id in new_ids else 'selected_history'
            payload[section][record_id] = item
        return payload

    @staticmethod
    def sent_snippets(memory, record_id):
        result = []
        for values in memory.entity_index.values():
            for value, refs in values.items():
                if not any(ref['record_id'] == record_id for ref in refs):
                    continue
                for sent in memory.records[record_id]['sent_versions']:
                    for match in re.finditer(re.escape(value), sent):
                        snippet = sent[max(0, match.start() - 40):match.end() + 40]
                        if snippet not in result:
                            result.append(snippet)
        return result
