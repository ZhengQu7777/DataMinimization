"""主流程：复用抽取 → 选择材料 → 复核策略 → 生成待发送快照。"""

from .privacy_memory import PrivacyMemory, fingerprint
from .router import Router
from .local_model import REVIEW_PROMPT
from copy import deepcopy


class PrivacyGuard:
    def __init__(self, task, model):
        self.model = model
        self.memory = PrivacyMemory(task)
        self.router = Router()
        self.reviews = []
        self._review_cache = {}

    def prepare(self, cloud_response, tool_results):
        """tool_results 每项含 tool_name、arguments、text；本函数不会发送数据。"""
        current_ids, new_ids, reused = [], [], 0
        for result in tool_results:
            name, args, text = result['tool_name'], result['arguments'], result['text']
            if not isinstance(text, str):
                raise TypeError('请传入工具实际返回文本；工具的格式由工具决定')
            record_id = self.memory.find(name, args, text)
            if record_id is None:
                summaries, mentions = [], []
                # 长返回分片抽取；相邻片段重叠 100 字符，降低实体被截断的概率。
                for start in range(0, max(1, len(text)), 2900):
                    analysis = self.model.extract(text[start:start + 3000])
                    summaries.append(analysis['summary'])
                    mentions.extend(analysis['entities'])
                record_id = self.memory.add(name, args, text, '\n'.join(summaries), mentions)
                new_ids.append(record_id)
            else:
                reused += 1
            self.memory.touch(record_id)
            current_ids.append(record_id)

        # 上次失败后留下的未判定实体也必须进入复核，不能直接返回原文。
        pending = [r for r, data in self.memory.records.items()
                   if set(data['entities']) - set(data['policies'])]
        selected = self.router.select(self.memory, cloud_response, list(dict.fromkeys(current_ids + pending)))
        payload = self.router.build_input(self.memory, cloud_response, selected, new_ids)
        decisions, payload = self._review(payload)
        requests = [d['context_request'] for d in decisions if d['decision'] == 'need_context']
        if requests:
            more = self.router.select(self.memory, cloud_response, current_ids + pending, '\n'.join(requests))
            selected = list(dict.fromkeys(selected + more))
            payload = self.router.build_input(self.memory, cloud_response, selected, new_ids, full_text=True)
            decisions, payload = self._review(payload)
            if any(d['decision'] == 'need_context' for d in decisions):
                raise RuntimeError('补读后仍缺少上下文，本轮没有可发送结果')

        # 全部输出校验通过后才修改策略，避免只更新了一半就发生错误。
        visible = {**payload['new_tool_results'], **payload['selected_history']}
        last_review = {
            'input_hash': fingerprint({'system_prompt': REVIEW_PROMPT, 'input': payload}),
            'read_ranges': {r: [[s['start'], s['start'] + len(s['text'])] for s in v['snippets']]
                            for r, v in visible.items()},
        }
        updates = []
        for decision in decisions:
            r, e = decision['record_id'], decision['entity_id']
            record = self.memory.records[r]
            if decision['decision'] == 'update':
                record['policies'][e] = {k: decision[k] for k in ['action', 'replacement']}
                record['sanitized_text'] = None
                self.memory.index(r, e, decision['replacement'])
                updates.append({'record_id': r, 'entity_id': e, 'action': decision['action']})
            record['policies'][e]['last_review'] = last_review
        texts = {r: self.memory.render(r) for r in self.memory.records}
        return {'texts': texts, 'current_record_ids': current_ids, 'reused': reused,
                'reviewed_record_ids': selected, 'updates': updates}

    def _review(self, payload):
        """格式或引用错误只纠正一次；始终保留真实调用输入和失败输出。"""
        records = {**payload['new_tool_results'], **payload['selected_history']}
        if not any(record['entities'] for record in records.values()):
            return [], payload
        cache_key = fingerprint({'system_prompt': REVIEW_PROMPT, 'input': payload})
        if cache_key in self._review_cache:
            output, actual_input = self._review_cache[cache_key]
            self.reviews.append({'input': actual_input, 'output': output, 'cache_hit': True})
            return output['decisions'], actual_input
        for attempt in range(2):
            output = self.model.review(payload)
            entry = {'input': payload, 'output': output, 'cache_hit': False}
            self.reviews.append(entry)
            try:
                decisions = self._validate(payload, output)
            except (ValueError, KeyError, TypeError) as exc:
                entry['error'] = str(exc)
                if attempt == 1:
                    raise
                payload = deepcopy(payload)
                payload['format_feedback'] = f'上次输出校验失败：{exc}。请纠正并返回全部实体；不要引用summary。'
                payload['previous_output'] = output
                continue
            if not any(d['decision'] == 'need_context' for d in decisions):
                self._review_cache[cache_key] = (output, payload)
            return decisions, payload

    @staticmethod
    def _validate(payload, output):
        """只检查结构与逐字引用；不把这些检查当作隐私判断正确性的证明。"""
        records = {**payload['new_tool_results'], **payload['selected_history']}
        expected = {(r, e) for r, data in records.items() for e in data['entities']}
        decisions, seen = output['decisions'], set()
        for d in decisions:
            ref = (d['record_id'], d['entity_id'])
            if ref not in expected or ref in seen:
                raise ValueError(f'模型返回了未知或重复实体：{ref}')
            seen.add(ref)
            entity = records[ref[0]]['entities'][ref[1]]
            if d['decision'] == 'need_context':
                if not d.get('context_request'):
                    raise ValueError('need_context 必须说明缺少什么')
                continue
            if d['decision'] == 'unchanged':
                if entity['current_policy'] is None:
                    raise ValueError('新实体不能使用 unchanged')
            elif d['decision'] == 'update':
                if d['action'] not in {'keep', 'replace', 'generalize', 'delete'}:
                    raise ValueError('未知处理策略')
                if not isinstance(d['replacement'], str):
                    raise ValueError('replacement 必须是字符串')
                if d['action'] == 'keep' and d['replacement'] != entity['value']:
                    raise ValueError('keep 必须保留原值')
                if d['action'] == 'delete' and d['replacement'] != '':
                    raise ValueError('delete 的 replacement 必须为空')
                if d['action'] in {'replace', 'generalize'} and not d['replacement']:
                    raise ValueError('替换或概括必须给出非空 replacement')
                if d['action'] in {'replace', 'generalize'} and d['replacement'] == entity['value']:
                    raise ValueError('替换或概括不能等于原值；保留原值应使用 keep')
            else:
                raise ValueError('未知复核结果')
            source, quote = d['evidence']['source'], d['evidence']['quote']
            evidence = {'task': [payload['task']], 'cloud': [payload['cloud_response'].get('content', '')]}
            evidence.update({r: [s['text'] for s in v['snippets']] for r, v in records.items()})
            if not quote or not any(quote in text for text in evidence.get(source, [])):
                raise ValueError(f'实体{ref}的证据无效：source={source!r}；只允许task/cloud/记录编号，quote须复制对应原文，不能引用摘要')
        if seen != expected:
            raise ValueError('模型遗漏了待判断实体，本轮没有可发送结果')
        return decisions
