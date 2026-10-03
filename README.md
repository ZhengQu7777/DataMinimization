# source 模块说明

## 1. 总览

| 文件 | 职责 |
| --- | --- |
| [privacy_memory.py](privacy_memory.py) | 存储工具原文、实体和策略，查重与文本替换 |
| [local_model.py](local_model.py) | 加载本地模型，执行实体抽取和策略判断 |
| [router.py](router.py) | 选择复核记录，构造模型输入 |
| [privacy_guard.py](privacy_guard.py) | 管理整轮处理、校验和策略更新 |
| [demo.py](demo.py) | 执行三轮合成交互，默认在终端展示结果，按需保存 |
| [test_privacy_guard.py](test_privacy_guard.py) | 使用预设模型结果测试程序逻辑 |

本文统一使用以下示例：

- 任务：`将联系人资料整理成匿名摘要。`
- 工具：`read_note`，参数：`{"note_id": 42}`，返回：`联系人：张三。`
- 假设模型识别出姓名“张三”，决定替换为“某联系人”。模型输出仅用于说明格式。

`touch()` 等方法返回 `None`，通过修改存储生效，具体变化列在职责栏；`find()` 返回 `None` 则表示没有找到重复记录。

## 2. privacy_memory.py

类：`PrivacyMemory`。`fingerprint()` 是文件中的独立函数。

### 属性

| 属性 | 示例值 | 内容与用途 |
| --- | --- | --- |
| `task` | `"将联系人资料整理成匿名摘要。"` | 用户最初的任务描述 |
| `records` | `{"1": 下面的记录字典}` | 按记录编号保存工具返回、摘要、实体和策略 |
| `_lookup` | `{("c51b70a2…", "59d6d128…"): "1"}` | 两个哈希分别对应工具调用和返回文本；相同组合可查到记录 `"1"` |
| `entity_index` | `{"PERSON": {"张三": [{"record_id": "1", "entity_id": "0"}]}}` | 姓名“张三”出现在记录 `"1"` 的实体 `"0"` 中；新代称也登记在这里 |
| `recent_ids` | `["2", "3", "1"]` | 记录 `"1"` 最近返回，`"3"` 次之；供路由选择最近出现的旧记录 |

哈希值只展示前 8 位。顺序列表的例子假设已有三条记录，其余例子只使用记录 `"1"`。下面是它首次处理后、尚未发送时的关键字段：

```json
{
  "tool_name": "read_note",
  "arguments": {"note_id": 42},
  "text": "联系人：张三。",
  "summary": "记录一名联系人的姓名。",
  "entities": {
    "0": {"type": "PERSON", "start": 4, "end": 6}
  },
  "policies": {
    "0": {"action": "replace", "replacement": "某联系人"}
  },
  "sent_versions": []
}
```

`entities["0"]` 和 `policies["0"]` 对应同一处姓名。`text[4:6]` 为“张三”：从第 4 个索引取到第 6 个索引之前，索引从 0 开始。

刚调用 `add()` 时，`policies` 还是 `{}`，复核后才写入策略。同名实体分别保存，不自动共享策略。

### 方法

| 方法 | 输入示例 | 返回示例 | 职责 |
| --- | --- | --- | --- |
| `__init__()` | `task="将联系人资料整理成匿名摘要。"` | `None` | 保存任务；记录和索引初始化为 `{}`，顺序列表初始化为 `[]` |
| `find()` | `tool_name="read_note"`；`arguments={"note_id":42}`；`text="联系人：张三。"` | 已保存则返回 `"1"`，未保存则返回 `None` | 判断同一个工具是否曾以相同参数返回完全相同的文本 |
| `add()` | 工具名、参数、原文同上一行；`summary="记录一名联系人的姓名。"`；`mentions=[{"value":"张三","type":"PERSON"}]` | 新建第一条记录时返回 `"1"` | 保存材料，并将“张三”定位为 `start=4, end=6` |
| `touch()` | `record_id="1"` | `None` | 顺序由 `["1","2","3"]` 变为 `["2","3","1"]`，表示记录 `"1"` 最近再次返回 |
| `index()` | `record_id="1"`；`entity_id="0"`；`value="某联系人"` | `None` | 在姓名索引中增加 `"某联系人": [{"record_id":"1","entity_id":"0"}]`，方便后续找回材料 |
| `render()` | `record_id="1"` | `"联系人：某联系人。"` | 按已有策略处理实体；策略未变时复用已生成的文本 |
| `mark_sent()` | `texts={"1":"联系人：某联系人。"}` | `None` | 记录 `"1"` 的发送历史从 `[]` 变为 `["联系人：某联系人。"]`；相同版本只保存一次 |
| `as_dict()` | 无参数；以刚初始化时为例 | `{"task":"将联系人资料整理成匿名摘要。","records":{},"entity_index":{}}` | 导出任务、记录和实体索引；有记录时包含已存的数据 |
| `fingerprint()` | `value={"note_id":42}` | `"69862a99…"`（实际为 64 位字符串） | 为内容生成哈希值，供程序比较内容是否相同 |

`touch()` 按工具返回更新顺序；读取旧记录进行复核不会改变这个顺序。

`render()` 在实体缺少策略时报错，无实体时保留原文。`mark_sent()` 只登记发送记录，实际请求由调用方发送。

## 3. local_model.py

类：`LocalModel`。`EXTRACT_PROMPT` 是实体抽取提示词，`REVIEW_PROMPT` 是策略判断提示词。

### 属性

| 属性 | 示例 | 内容与用途 |
| --- | --- | --- |
| `tokenizer` | Qwen3 模型配套的分词器对象 | 将文字转换为模型输入，并将模型输出转换回文字 |
| `model` | 从 `models/Qwen3-4B-Instruct-2507` 加载的模型对象 | 执行本地推理 |
| `calls` | `[{"kind":"extract","elapsed_seconds":0.8}]`（日志节选，耗时示意） | 完整日志还保存实际输入消息和原始模型输出 |

### 方法

| 方法 | 输入示例 | 返回示例 | 职责 |
| --- | --- | --- | --- |
| `__init__()` | `model_path="models/Qwen3-4B-Instruct-2507"`；`device="cuda:0"` | `None` | 加载本地模型，准备调用日志 |
| `_ask()` | `kind="extract"`；`system=EXTRACT_PROMPT`；`payload={"text":"联系人：张三。"}` | 例如下方的抽取结果字典 | 将提示词和材料发送给本地模型，解析回答并记录本次调用 |
| `extract()` | `text="联系人：张三。"` | 下方含 `summary` 和 `entities` 的字典 | 生成摘要，识别原文中的 PII，去除重复项 |
| `review()` | `payload`：第 4 节的首次复核输入 | 下方含 `decisions` 的字典 | 为每个实体提出处理策略，并比较新旧策略是否相同 |

### 输出结构

输入 `text="联系人：张三。"`，假设 `extract()` 返回：

```json
{
  "summary": "记录一名联系人的姓名。",
  "entities": [{"value": "张三", "type": "PERSON"}]
}
```

`entities` 包含原值和类型；`privacy_memory.py` 中的 `add()` 负责确定字符位置。抽取阶段不决定处理策略。

首次复核时尚无旧策略。假设模型决定替换姓名，`review()` 返回：

```json
{
  "decisions": [{
    "record_id": "1",
    "entity_id": "0",
    "action": "replace",
    "replacement": "某联系人",
    "reason": "匿名摘要不需要真实姓名。",
    "evidence_id": "1.0",
    "decision": "update",
    "evidence": {"source": "1", "quote": "张三"}
  }]
}
```

模型提出 `action`、`replacement`、`reason` 和 `evidence_id`；程序补充 `decision` 和 `evidence`。例如 `"1.0"` 指记录 `"1"` 的实体 `"0"`，程序据此取出原文“张三”。

- 首次没有策略，或新旧策略不同：`decision="update"`；已有同样的替换策略：`decision="unchanged"`。
- `keep` 保留原值，`delete` 删除，`replace` 使用代称，`generalize` 使用更概括的表述。
- 材料不足时可返回 `action="need_context"`，并给出 `context_request`，例如 `"需要查看邮件的投递结果。"`。

`review()` 只返回决定，保存新策略由 `privacy_guard.py` 负责。

`extract()` 每次接收最多 3000 字符。`_ask()` 将提示词作为系统消息、输入 JSON 作为用户消息；输入超过 8192 token，或输出达到 2048 token 且未结束时，抛出异常。

## 4. router.py

类：`Router`。`words()` 是文件中的独立函数。路由由程序完成，不调用语言模型。

### 属性

| 属性 | 示例值 | 内容与用途 |
| --- | --- | --- |
| `history_limit` | `4` | 每次额外选取的旧记录数量上限 |
| `_terms` | `{"1": {"read_note", "note_id", "42", "联系", "系人", "张三"}}` | 保存工具名、参数和原文中的词，供检索比较；内层是 Python 集合 |

### 方法

| 方法 | 输入示例 | 返回示例 | 职责 |
| --- | --- | --- | --- |
| `__init__()` | `history_limit=4` | `None` | 保存上限，初始化词集合缓存 |
| `select()` | `memory`：仅保存记录 `"1"` 的 `PrivacyMemory`；`cloud_response={"content":"开始整理匿名摘要。"}`；`current_ids=["1"]`；`extra_query=""` | `["1"]` | 确定本轮需要检查哪些记录的处理策略 |
| `build_input()` | `memory`、`cloud_response` 同上一行；`selected_ids=["1"]`；`new_ids=["1"]`；`full_text=False` | 下方的首次复核输入字典 | 将任务、执行进展和选中的材料整理成本地模型输入 |
| `sent_snippets()` | `memory`：已登记发送版本的隐私记忆；`record_id="1"` | `["联系人：某联系人。"]`；尚未登记发送时为 `[]` | 找出这份材料曾向云端披露过的实体或代称及其周围文本 |
| `words()` | `text="张三 Read_Note"` | `{"张三", "read_note"}` | 提取小写英文/数字词及相邻两个汉字 |

`current_ids` 包含本轮记录及仍有实体缺少策略的记录。

选择旧记录时，先合并三类候选：

1. 实体原值或代称出现在本轮内容中的记录。
2. 最近返回过的两条旧记录。
3. 与本轮内容共同词最多的三条旧记录，共同词数量须大于零。

合并后优先选实体匹配的记录，再优先选最近返回的记录，其次比较共同词数量，最后用较大记录编号打破平局。默认最多选四条旧记录；必查记录全部保留。本轮检索内容由云端响应、必查原文和补读请求组成。

### 首次复核输入

此时已保存记录 `"1"`，尚未判断策略，也未发送。`build_input()` 返回的字典就是 `review()` 接收的 `payload`：

```json
{
  "task": "将联系人资料整理成匿名摘要。",
  "cloud_response": {"content": "开始整理匿名摘要。"},
  "new_tool_results": {
    "1": {
      "tool_name": "read_note",
      "arguments": {"note_id": 42},
      "summary": "记录一名联系人的姓名。",
      "snippets": [{"start": 0, "text": "联系人：张三。"}],
      "entities": {
        "0": {"type": "PERSON", "value": "张三", "current_policy": null}
      },
      "sent_snippets": []
    }
  },
  "selected_history": {}
}
```

`current_policy=null` 表示尚无策略。再次返回同一材料时，`new_ids=[]`，记录 `"1"` 放入 `selected_history`，并带上已有策略：`{"action":"replace","replacement":"某联系人"}`。

`snippets` 默认截取实体前后各 60 字符并合并重叠部分，本例原文短，所以全部保留。没有实体时取前 240 字符；`full_text=True` 时提供选中记录全文。

## 5. privacy_guard.py

类：`PrivacyGuard`。

### 属性

| 属性 | 示例 | 内容与用途 |
| --- | --- | --- |
| `model` | 已加载 Qwen3 的 `LocalModel` | 执行实体抽取和策略判断 |
| `memory` | 保存第 2 节记录 `"1"` 的 `PrivacyMemory` | 保存工具材料、实体和策略 |
| `router` | 设置 `history_limit=4` 的 `Router` | 选择复核材料并整理模型输入 |
| `reviews` | `[{"cache_hit": false}]`（日志节选） | 此次复核调用了模型；完整日志还含输入和输出，校验失败时记录错误 |
| `_review_cache` | 初始为 `{}`；复核成功后按输入哈希保存“结果字典、实际输入”二元组 | 提示词和输入相同时复用判断 |

### 方法

| 方法 | 输入示例 | 返回示例 | 职责 |
| --- | --- | --- | --- |
| `__init__()` | `task="将联系人资料整理成匿名摘要。"`；`model`：已加载的 `LocalModel` | `None` | 准备本地模型、隐私记忆、路由和复核日志 |
| `prepare()` | 云端响应为 `{"content":"开始整理匿名摘要。"}`，工具返回为本例联系人资料；完整参数见第 7 节 | 例如 `texts={"1":"联系人：某联系人。"}`，另含下表状态字段 | 完成查重、抽取、路由、复核、校验、策略更新和文本生成 |
| `_review()` | `payload`：第 4 节的首次复核输入 | `(decisions, actual_input)`：第 3 节的决定列表，以及本次实际输入；未纠错时输入不变 | 复用已有判断，或调用本地模型判断并校验结果 |
| `_validate()` | `payload`：第 4 节输入；`output`：第 3 节的 `review()` 返回字典 | `output["decisions"]`，即只返回其中的列表 | 检查是否遗漏实体、策略是否符合格式要求、引用是否来自提供的原文 |

例如把输出中的 `entity_id` 改成不存在的 `"9"`，`_validate()` 会报错，不返回决定列表。

### prepare 输出

| 字段 | 首次调用的示例值 | 内容 |
| --- | --- | --- |
| `texts` | `{"1":"联系人：某联系人。"}` | 全部已存记录的处理后文本 |
| `current_record_ids` | `["1"]` | 本轮工具返回对应的记录编号 |
| `reused` | `0` | 本轮重复材料数；下轮只重复返回这一条时为 `1` |
| `reviewed_record_ids` | `["1"]` | 选入复核的记录编号；不代表一定调用了模型 |
| `updates` | `[{"record_id":"1","entity_id":"0","action":"replace"}]` | 本轮改变策略的实体；没有变化时为 `[]` |

`_review()` 在没有实体时直接跳过判断。提示词和输入相同时复用已有结果，否则调用本地模型。输出校验失败时要求模型纠正一次；模型调用或解析报错则直接中止。

`prepare()` 最多补充上下文一次。所有决定通过校验后才更新策略；失败前新建的记录可能保留，其中仍有实体缺少策略的记录会在下轮必查。未选中的旧记录沿用已有策略。上述校验不能证明模型的隐私判断一定正确。

## 6. demo.py 与 test_privacy_guard.py

以下预设模型只识别“林舟”“周宁”，因此测试例子使用“林舟”。它不会把前文的“张三”识别为实体。

| 方法或函数 | 输入示例 | 返回示例与职责 |
| --- | --- | --- |
| `demo.py`：`main()` | 无函数参数；命令行例：`--device cuda:0 --output experiments/privacy_demo/manual` | `None`；加载真实本地模型，处理文件中三轮合成材料；默认仅终端展示，指定 `--output` 才保存结果 |
| `test_privacy_guard.py`：`tool()` | `text="姓名：林舟。"`；`number=1` | `{"tool_name":"read_record","arguments":{"id":1},"text":"姓名：林舟。"}`；包装工具返回 |
| `test_privacy_guard.py`：`decisions()` | `payload` 中有记录 `"1"`、实体 `"0"`“林舟”；`replacement="[匿名]"` | 决定列表，例：`[{"record_id":"1","entity_id":"0","action":"replace","replacement":"[匿名]"}]`（节选） |
| `FakeModel.__init__()` | `handler=None`；`summary="合成摘要"` | `None`；使用默认策略生成函数，并准备调用记录 |
| `FakeModel.extract()` | `text="姓名：林舟。"` | `{"summary":"合成摘要","entities":[{"value":"林舟","type":"PERSON"}]}`；返回预设结果 |
| `FakeModel.review()` | 含“林舟”的复核输入；未设置自定义 `handler` | 将 `decisions()` 的列表放入 `{"decisions": [...]}` 返回；不运行语言模型 |
| `PrivacyGuardTests` 的 `test_*()` | 无参数，由 unittest 调用 | 正常返回 `None`；例如检查重复材料只抽取一次，不满足预期则测试失败 |

演示采用合成的云端响应和工具返回。默认不生成文件；指定 `--output` 后，`memory.json` 保存隐私记忆，`reviews.json` 保存复核记录，`model_calls.json` 保存本地模型调用日志，`summary.md` 保存演示报告。逐阶段数据格式可查看 [JSON 示例](../experiments/privacy_demo/json_walkthrough.json)。

## 7. 调用流程

### 7.1 初始化与输入

`demo.py` 先加载本地模型，再用任务描述和模型创建 `PrivacyGuard`。每轮调用 `prepare()`，传入云端响应和本轮工具返回。

`prepare()` 的输入示例：

```json
{
  "cloud_response": {"content": "开始整理匿名摘要。"},
  "tool_results": [
    {"tool_name": "read_note", "arguments": {"note_id": 42}, "text": "联系人：张三。"}
  ]
}
```

`tool_results` 的每项对应一次完整工具返回。外围字典为插件输入格式，`text` 保存工具原文字符串。

### 7.2 执行步骤

整个流程由 `privacy_guard.py` 中的 `prepare()` 组织：

| 步骤 | 文件与方法 | 执行逻辑 |
| --- | --- | --- |
| 1. 查重复 | `privacy_memory.py`：`find()` | 比较工具名、参数和完整返回文本，查找是否已有相同记录 |
| 2. 抽取新材料 | `local_model.py`：`extract()` | 新材料生成摘要并抽取 PII；重复材料直接复用已有抽取结果 |
| 3. 保存记录 | `privacy_memory.py`：`add()`、`touch()` | 新材料保存为一条记录；无论新旧，都把本轮返回的记录编号放到顺序列表末尾 |
| 4. 选择复核材料 | `router.py`：`select()` | 选取本轮记录、仍缺策略的记录，以及少量可能相关的旧记录 |
| 5. 整理模型输入 | `router.py`：`build_input()` | 合并任务、云端响应、选中材料的原文片段、摘要、实体、已有策略和已发送片段 |
| 6. 判断并校验 | `privacy_guard.py`：`_review()`、`_validate()`；`local_model.py`：`review()` | 复用已有判断，或调用模型为实体提出策略，再检查输出是否完整、引用是否有效 |
| 7. 必要时补读 | `router.py`：`select()`、`build_input()`；`privacy_guard.py`：`_review()` | 模型请求更多上下文时，再找相关记录，补充选中材料全文后重判一次；仍无法判断则报错 |
| 8. 更新策略 | `privacy_guard.py`：`prepare()`；`privacy_memory.py`：`index()` | 全部校验通过后保存新策略，并登记新代称，供后续查找相关材料 |
| 9. 生成文本 | `privacy_memory.py`：`render()` | 为全部已存记录生成处理后的文本；策略未变的记录复用已有文本 |

长材料在抽取前按 3000 字符分片，相邻片段重叠 100 字符。重复材料跳过抽取，但仍参与本轮策略复核。

### 7.3 输出与发送

首次调用的返回示例，假设姓名替换为“某联系人”：

```json
{
  "texts": {"1": "联系人：某联系人。"},
  "current_record_ids": ["1"],
  "reused": 0,
  "reviewed_record_ids": ["1"],
  "updates": [{"record_id": "1", "entity_id": "0", "action": "replace"}]
}
```

调用方从 `texts` 选择实际要发送的文本，组装云端请求。发送后将这些文本交给 `privacy_memory.py` 中的 `mark_sent()` 登记。当前 `demo.py` 只模拟登记，不实际调用云端。

重复返回复用抽取结果，策略复核仍取决于本轮输入和缓存。更新本地策略不会撤回已发送的信息。
