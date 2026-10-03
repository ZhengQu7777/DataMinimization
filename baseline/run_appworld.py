"""DeepSeek Flash + LangGraph 跑一条 AppWorld dev 任务，不过滤请求。

运行：python -u source/baseline/run_appworld.py [task_id]
详细输出自动保存到 experiments/logs/。
本地计时排除日志写入/刷新以及模型和 AppWorld 的 HTTP 调用，PII 初始化单列。
"""

import argparse
import json
import os
import sys
import time
import traceback
import uuid
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

import requests
from langgraph.graph import END, START, StateGraph

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.getenv("APPWORLD_ROOT", PROJECT_ROOT / "datasets/appworld"))
WORLD_URL = os.getenv("APPWORLD_URL", "http://127.0.0.1:9000").rstrip("/")
MODEL_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"
API_KEY = "sk-3a071c42ea3f4517a6e5fe7d425d0ad1"
MAX_ROUNDS, MAX_EXECUTIONS = 40, 50

PROMPT = """Solve the AppWorld task using execute_python. Variables persist; print results.
The apis object is already available in this environment; do not import apis.
Discover APIs via apis.api_docs: show_app_descriptions(),
show_api_descriptions(app_name=...), show_api_doc(app_name=..., api_name=...).
Use supervisor APIs for account details and handle pagination.
Access only simulated apps, never host files or ground truth.
Finish with apis.supervisor.complete_task(answer=...) for questions,
or apis.supervisor.complete_task() for actions.
"""
TOOL = {"type": "function", "function": {
    "name": "execute_python",
    "description": "Run Python in the persistent AppWorld environment. Print results to inspect them.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                   "required": ["code"], "additionalProperties": False},
}}


class LocalClock:
    """Pause for HTTP and log I/O; nested pauses are excluded only once."""

    def __init__(self):
        self.excluded = 0.0
        self.pause_depth = 0
        self.paused_at = 0.0

    def now(self):
        current = self.paused_at if self.pause_depth else time.perf_counter()
        return current - self.excluded

    @contextmanager
    def pause(self):
        if self.pause_depth == 0:
            self.paused_at = time.perf_counter()
        self.pause_depth += 1
        try:
            yield
        finally:
            self.pause_depth -= 1
            if self.pause_depth == 0:
                self.excluded += time.perf_counter() - self.paused_at


class TimedLog:
    """Preserve stdout/stderr output without charging write/flush to processing."""

    def __init__(self, stream, clock):
        self.stream = stream
        self.clock = clock

    def write(self, text):
        with self.clock.pause():
            return self.stream.write(text)

    def flush(self):
        with self.clock.pause():
            return self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def post(session, url, **payload):
    response = session.post(url, json=payload, timeout=(15, 180))
    if not response.ok:
        raise RuntimeError(f"{url}: HTTP {response.status_code}\n{response.text[:1500]}")
    return response.json()


def main(task_id=None, masker=None, clock=None):
    clock = clock if clock is not None else LocalClock()
    if masker is not None and hasattr(masker, "set_clock"):
        masker.set_clock(clock)
    with redirect_stdout(TimedLog(sys.stdout, clock)), redirect_stderr(TimedLog(sys.stderr, clock)):
        return _run_task(task_id, masker, clock)


def _run_task(task_id, masker, clock):
    task_ids = [s.strip() for s in (ROOT / "data/datasets/dev.txt").read_text().splitlines()
                if s.strip() and not s.lstrip().startswith("#")]
    task_id = task_id or task_ids[0]
    if task_id not in task_ids:
        raise SystemExit(f"{task_id} 不在当前 dev.txt 中")
    variant = getattr(masker, "variant", "fullmask") if masker is not None else "raw"
    experiment = f"deepseek_{variant}_" + uuid.uuid4().hex[:10]
    stats = dict(rounds=0, executions=0, prompt_tokens=0, completion_tokens=0,
                 local_seconds=0.0, pii_seconds=0.0)
    started = clock.now()

    def call_api(session, url, **payload):
        """Exclude the whole HTTP call, including tool execution and API errors."""
        with clock.pause():
            return post(session, url, **payload)

    with requests.Session() as app, requests.Session() as llm:
        app.trust_env = False  
        llm.headers["Authorization"] = "Bearer " + API_KEY

        def world(route, **kwargs):
            return call_api(app, f"{WORLD_URL}/{route}", task_id=task_id, **kwargs)["output"]

        def advance(state):
            stats["rounds"] += 1
            print(f"\n[第 {stats['rounds']} 轮]", flush=True)
            request_messages = state["messages"]
            if masker is not None:
                pii_started = clock.now()
                try:
                    request_messages = masker.mask_messages(state["messages"])
                finally:
                    stats["pii_seconds"] += clock.now() - pii_started
                if hasattr(masker, "last_counts"):
                    print("[PII 缓存]", masker.last_counts, flush=True)
                if hasattr(masker, "api_seconds"):
                    print(f"[脱敏 API] 本轮: {masker.last_api_seconds:.6f}s | "
                          f"累计: {masker.api_seconds:.6f}s | token: {masker.last_usage}", flush=True)
            data = call_api(llm, MODEL_URL, model=MODEL, messages=request_messages,
                        tools=[TOOL], tool_choice="auto", thinking={"type": "disabled"},
                        temperature=0, max_tokens=2048, stream=False)
            for name in ("prompt_tokens", "completion_tokens"):
                stats[name] += data.get("usage", {}).get(name, 0)
            choice = data["choices"][0]
            if choice["finish_reason"] == "length":
                raise RuntimeError("模型输出被截断，本轮未执行。")
            message = {k: v for k, v in choice["message"].items()
                       if k in ("role", "content", "tool_calls", "reasoning_content") and v is not None}
            messages = state["messages"] + [message]
            print(message.get("content", ""), flush=True)
            calls, done = message.get("tool_calls", []), False
            for call in calls:  # 串行执行，同一任务保留变量和数据库状态。
                if done or stats["executions"] >= MAX_EXECUTIONS:
                    output = "Skipped: task ended or execution budget exhausted."
                else:
                    try:
                        if call["function"]["name"] != "execute_python":
                            raise ValueError("Use execute_python.")
                        code = json.loads(call["function"]["arguments"])["code"]
                        if not isinstance(code, str):
                            raise ValueError("code must be a string.")
                    except (ValueError, KeyError, TypeError) as exc:
                        output = f"Invalid tool arguments: {exc}"
                    else:
                        stats["executions"] += 1
                        print(f"[执行]\n{code}", flush=True)
                        output = world("execute", code=code)
                        done = world("task_completed")
                print(f"[观察]\n{output}", flush=True)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": str(output)})
            stop = done or not calls or stats["rounds"] >= MAX_ROUNDS or stats["executions"] >= MAX_EXECUTIONS
            return {"messages": messages, "stop": stop}

        def step(state):
            step_started = clock.now()
            pii_before = stats["pii_seconds"]
            try:
                return advance(state)
            finally:
                local_seconds = max(0.0, clock.now() - step_started)
                print(f"[本地计时] 本轮: {local_seconds:.6f}s | "
                      f"其中 PII 过滤: {stats['pii_seconds'] - pii_before:.6f}s", flush=True)

        graph = StateGraph(dict)
        graph.add_node("step", step)
        graph.add_edge(START, "step")
        graph.add_conditional_edges("step", lambda s: END if s["stop"] else "step", [END, "step"])
        agent = graph.compile()

        task = world("initialize", experiment_name=experiment, max_interactions=MAX_EXECUTIONS)
        try:
            print(f"任务: {task_id} | 模型: {MODEL} | 实验: {experiment}")
            print(task["instruction"], flush=True)
            agent.invoke({"messages": [{"role": "system", "content": PROMPT},
                                       {"role": "user", "content": json.dumps(task, ensure_ascii=False)}]},
                         config={"recursion_limit": MAX_ROUNDS + 5})
        finally:
            try:
                score = world("evaluate", report=False, suppress_errors=True)
                print(f"\n官方 success: {score['success']} | 检查通过: {len(score['passes'])}/{score['num_tests']}")
                if score["failures"]:
                    print("失败项:", json.dumps(score["failures"], ensure_ascii=False, indent=2))
                print("输出:", ROOT / "experiments/outputs" / experiment / "tasks" / task_id)
            finally:
                try:
                    world("close")
                finally:
                    stats["local_seconds"] = max(0.0, clock.now() - started)
                    if masker is not None and hasattr(masker, "api_seconds"):
                        stats["sanitizer_api_seconds"] = masker.api_seconds
                        stats["sanitizer_api_calls"] = masker.api_calls
                    print("统计:", {k: round(v, 6) if isinstance(v, float) else v
                                    for k, v in stats.items()})
                    print(f"[本地计时] 任务累计（不含初始化、日志 I/O 和接口调用）: {stats['local_seconds']:.6f}s | "
                          f"其中 PII 过滤: {stats['pii_seconds']:.6f}s", flush=True)
    return score["success"]


def run(masker_factory=None):
    parser = argparse.ArgumentParser(description="运行一条 AppWorld dev 任务。")
    parser.add_argument("task_id", nargs="?", help="任务 ID，默认 dev.txt 中的第一条")
    args = parser.parse_args()
    variant = getattr(masker_factory, "variant", "fullmask") if masker_factory is not None else "raw"
    log_dir = PROJECT_ROOT / "experiments/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"appworld_{variant}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.log"
    console = sys.stdout
    print(f"任务运行中，日志: {log_path}", flush=True)
    clock = LocalClock()
    with log_path.open("x", encoding="utf-8", buffering=1) as log, \
            redirect_stdout(TimedLog(log, clock)), redirect_stderr(TimedLog(log, clock)):
        try:
            init_started = clock.now()
            masker = masker_factory() if masker_factory is not None else None
            init_seconds = clock.now() - init_started if masker_factory is not None else 0.0
            print(f"[本地初始化] PII 过滤器: {init_seconds:.6f}s", flush=True)
            success = main(task_id=args.task_id, masker=masker, clock=clock)
        except BaseException:
            traceback.print_exc()
            print("运行出错或已中断，详情见日志。", file=console, flush=True)
            raise SystemExit(1) from None
    print(f"运行结束，官方 success: {success}")


if __name__ == "__main__":
    run()
