"""离线契约测试；假模型只生成固定决定，不验证模型的隐私判断能力。"""

import copy
import unittest

from source.local_model import LocalModel
from source.privacy_guard import PrivacyGuard


def tool(text, number=1):
    return {'tool_name': 'read_record', 'arguments': {'id': number}, 'text': text}


def decisions(payload, replacement='[匿名]'):
    records = {**payload['new_tool_results'], **payload['selected_history']}
    return [
        {'record_id': r, 'entity_id': e, 'decision': 'update',
         'action': 'replace', 'replacement': replacement, 'reason': '合成测试决定',
         'evidence': {'source': r, 'quote': entity['value']}}
        for r, record in records.items() for e, entity in record['entities'].items()
    ]


class FakeModel:
    def __init__(self, handler=None, summary='合成摘要'):
        self.handler = handler
        self.summary = summary
        self.extractions = []
        self.reviews = []

    def extract(self, text):
        self.extractions.append(text)
        return {'summary': self.summary, 'entities': [
            {'value': name, 'type': 'PERSON'}
            for name in ('林舟', '周宁') if name in text
        ]}

    def review(self, payload):
        self.reviews.append(copy.deepcopy(payload))
        result = self.handler(payload) if self.handler else decisions(payload)
        return {'decisions': result}


class PrivacyGuardTests(unittest.TestCase):
    def test_repeated_return_reuses_extraction_but_changed_cloud_can_update(self):
        model = FakeModel(lambda p: decisions(p, p['cloud_response']['content']))
        guard = PrivacyGuard('处理合成材料', model)
        result = tool('姓名：林舟。')
        first = guard.prepare({'content': '[核对阶段]'}, [result])
        second = guard.prepare({'content': '[提醒阶段]'}, [result])
        self.assertEqual(len(model.extractions), 1)
        self.assertEqual(first['current_record_ids'], second['current_record_ids'])
        self.assertEqual(second['reused'], 1)
        self.assertEqual(first['texts']['1'], '姓名：[核对阶段]。')
        self.assertEqual(second['texts']['1'], '姓名：[提醒阶段]。')
        self.assertTrue(second['updates'])

    def test_same_name_in_different_records_does_not_share_policy(self):
        def review(payload):
            result = decisions(payload)
            for item in result:
                if payload['cloud_response']['content'] == '初次':
                    item['replacement'] = '[甲]' if item['record_id'] == '1' else '[乙]'
                elif item['record_id'] == '1':
                    item['replacement'] = '[第一条新策略]'
                else:
                    item['decision'] = 'unchanged'
            return result

        guard = PrivacyGuard('两个同名的合成人物', FakeModel(review))
        guard.prepare({'content': '初次'}, [tool('甲组：林舟。'), tool('乙组：林舟。', 2)])
        result = guard.prepare({'content': '只调整第一条'}, [])
        self.assertEqual(result['texts']['1'], '甲组：[第一条新策略]。')
        self.assertEqual(result['texts']['2'], '乙组：[乙]。')
        self.assertEqual(guard.memory.records['2']['policies']['0']['replacement'], '[乙]')

    def test_need_context_receives_original_text_before_update(self):
        text = '姓名：林舟。' + '无关说明。' * 60 + '原文末尾依据'
        calls = []

        def review(payload):
            calls.append(payload)
            result = decisions(payload)
            for item in result:
                if len(calls) == 1:
                    item.update(decision='need_context', context_request='需要原文末尾依据')
                else:
                    item['evidence'] = {'source': '1', 'quote': '原文末尾依据'}
            return result

        model = FakeModel(review)
        guard = PrivacyGuard('读取合成说明', model)
        guard.prepare({'content': '检查'}, [tool(text)])
        self.assertEqual(len(model.reviews), 2)
        initial = model.reviews[0]['new_tool_results']['1']['snippets']
        expanded = model.reviews[1]['new_tool_results']['1']['snippets']
        self.assertFalse(any('原文末尾依据' in s['text'] for s in initial))
        self.assertTrue(any(s['text'] == text for s in expanded))
        self.assertEqual(guard.memory.records['1']['policies']['0']['replacement'], '[匿名]')

    def test_summary_cannot_be_used_as_original_evidence(self):
        def review(payload):
            result = decisions(payload)
            result[0]['evidence'] = {'source': '1', 'quote': '仅摘要中存在的说法'}
            return result

        model = FakeModel(review, summary='仅摘要中存在的说法')
        guard = PrivacyGuard('检查证据', model)
        with self.assertRaisesRegex(ValueError, '证据'):
            guard.prepare({'content': '检查'}, [tool('姓名：林舟。')])
        self.assertEqual(guard.memory.records['1']['policies'], {})
        self.assertEqual(guard.memory.records['1']['sent_versions'], [])

    def test_invalid_review_does_not_partially_update_existing_policies(self):
        for failure in ('missing_entity', 'unknown_ref', 'persistent_need_context'):
            with self.subTest(failure=failure):
                model = FakeModel()
                guard = PrivacyGuard('核对合成人名', model)
                result = tool('姓名：林舟、周宁。')
                guard.prepare({'content': '初始化'}, [result])
                previous = copy.deepcopy(guard.memory.records['1']['policies'])

                def review(payload):
                    items = decisions(payload, '[不能部分提交]')
                    if failure == 'missing_entity':
                        return items[:1]
                    if failure == 'unknown_ref':
                        items[1]['record_id'] = '不存在的记录'
                    else:
                        items[1].update(decision='need_context', context_request='继续补读')
                    return items

                model.handler = review
                error = RuntimeError if failure == 'persistent_need_context' else ValueError
                with self.assertRaises(error):
                    guard.prepare({'content': failure}, [result])
                self.assertEqual(guard.memory.records['1']['policies'], previous)
                self.assertEqual(guard.memory.records['1']['sent_versions'], [])

    def test_mark_sent_records_supplied_snapshot_not_latest_preparation(self):
        model = FakeModel(lambda p: decisions(p, p['cloud_response']['content']))
        guard = PrivacyGuard('检查发送快照', model)
        result = tool('姓名：林舟。')
        first = guard.prepare({'content': '[版本A]'}, [result])
        second = guard.prepare({'content': '[版本B]'}, [result])
        record = guard.memory.records['1']
        self.assertEqual(record['sent_versions'], [])
        self.assertNotEqual(first['texts'], second['texts'])
        guard.memory.mark_sent(first['texts'])
        self.assertEqual(record['sent_versions'], [first['texts']['1']])
        self.assertNotIn(second['texts']['1'], record['sent_versions'])
        self.assertEqual(record['sanitized_text'], second['texts']['1'])

    def test_review_receives_tool_metadata_for_new_and_historical_records(self):
        model = FakeModel()
        guard = PrivacyGuard('关联信息可能只在参数中', model)
        first = tool('姓名：林舟。')
        first['arguments'] = {'contact_ref': 'C4'}
        guard.prepare({'content': '第一轮'}, [first])
        second = tool('姓名：周宁。', 2)
        guard.prepare({'content': '第二轮'}, [second])
        payload = model.reviews[-1]
        for section, record_id, original in (
            ('selected_history', '1', first), ('new_tool_results', '2', second)
        ):
            record = payload[section][record_id]
            self.assertEqual(record['tool_name'], original['tool_name'])
            self.assertEqual(record['arguments'], original['arguments'])

    def test_repeated_record_is_recent_even_without_lexical_match(self):
        guard = PrivacyGuard('检查实际出现顺序', FakeModel())
        guard.router.history_limit = 2
        results = [tool(f'姓名：林舟；记录{i}。', i) for i in range(1, 7)]
        first = guard.prepare({'content': '初次'}, results)
        record_id = first['current_record_ids'][0]
        guard.prepare({'content': '再次读取'}, [results[0]])
        selected = guard.router.select(guard.memory, {'content': 'zxqv842'}, [])
        self.assertIn(record_id, selected)
        self.assertLessEqual(len(selected), 2)

    def test_caller_mutation_cannot_rewrite_stored_arguments_or_review_input(self):
        guard = PrivacyGuard('保存独立快照', FakeModel())
        result = tool('姓名：林舟。')
        result['arguments'] = {'recipient': {'ref': 'C4'}}
        cloud = {'content': '初始内容', 'tool_calls': [{'id': 'call-1'}]}
        arguments_before, cloud_before = copy.deepcopy(result['arguments']), copy.deepcopy(cloud)
        guard.prepare(cloud, [result])
        result['arguments']['recipient']['ref'] = 'C9'
        cloud['content'] = '后来修改'
        cloud['tool_calls'][0]['id'] = 'call-changed'
        self.assertEqual(guard.memory.records['1']['arguments'], arguments_before)
        saved = guard.reviews[0]['input']
        self.assertEqual(saved['cloud_response'], cloud_before)
        self.assertEqual(saved['new_tool_results']['1']['arguments'], arguments_before)

    def test_identical_complete_review_input_hits_cache(self):
        model = FakeModel()
        guard = PrivacyGuard('检查完整输入缓存', model)
        for _ in range(3):
            guard.prepare({'content': '相同进展'}, [tool('姓名：林舟。')])
        self.assertEqual(len(model.extractions), 1)
        self.assertEqual(len(model.reviews), 2)
        self.assertEqual([r['cache_hit'] for r in guard.reviews], [False, False, True])

    def test_invalid_quote_retries_once_before_committing_policy(self):
        def review(payload):
            self.assertEqual(guard.memory.records['1']['policies'], {})
            items = decisions(payload, '[合法结果]')
            if 'format_feedback' not in payload:
                items[0]['replacement'] = '[无效结果]'
                items[0]['evidence']['quote'] = '不存在于原文的证据'
            return items

        model = FakeModel(review)
        guard = PrivacyGuard('检查格式纠正', model)
        result = guard.prepare({'content': '检查'}, [tool('姓名：林舟。')])
        self.assertEqual(len(model.reviews), 2)
        self.assertIn('format_feedback', model.reviews[1])
        self.assertTrue(guard.reviews[0]['error'])
        self.assertNotIn('error', guard.reviews[1])
        self.assertEqual(result['texts']['1'], '姓名：[合法结果]。')

    def test_local_review_maps_evidence_and_compares_policy_without_mutating_output(self):
        fake = FakeModel()
        PrivacyGuard('检查协议转换', fake).prepare({'content': '检查'}, [tool('姓名：林舟。')])
        payload = fake.reviews[0]
        raw = {'decisions': [{'record_id': '1', 'entity_id': '0', 'action': 'replace',
                             'replacement': '[匿名]', 'reason': '测试', 'evidence_id': '1.0'}]}
        original = copy.deepcopy(raw)
        model = LocalModel.__new__(LocalModel)
        model._ask = lambda *args: raw
        for policy, expected in ((None, 'update'),
                                 ({'action': 'replace', 'replacement': '[匿名]'}, 'unchanged'),
                                 ({'action': 'delete', 'replacement': ''}, 'update')):
            payload['new_tool_results']['1']['entities']['0']['current_policy'] = policy
            item = model.review(payload)['decisions'][0]
            self.assertEqual(item['decision'], expected)
            self.assertEqual(item['evidence'], {'source': '1', 'quote': '林舟'})
            self.assertEqual(raw, original)


if __name__ == '__main__':
    unittest.main()
