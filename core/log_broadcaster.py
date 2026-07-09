"""日志中间件：把探索过程中的关键事件实时推送到本地网页前端

不改hello_agents库代码，而是给已经构造好的TraceLogger实例的log_event方法
包一层：原有的JSONL/HTML落盘行为照常执行，同时把同一份事件通过SSE广播给
所有连着的浏览器页面，运行时就能在前端看实时日志。
"""
from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime
from typing import Any, Callable

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, StreamingResponse
from starlette.routing import Route

_DASHBOARD_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>MAA+Agent 实时日志</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 20px;
    background: #12141a; color: #d8dee9;
    font-family: 'Consolas', 'Cascadia Code', monospace;
  }
  header {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 16px;
  }
  h1 { font-size: 16px; margin: 0; color: #7ee787; font-weight: 600; }
  #dot { width: 10px; height: 10px; border-radius: 50%; background: #6e7681; }
  #dot.live { background: #3fb950; box-shadow: 0 0 6px #3fb950; }
  #dot.dead { background: #f85149; }
  #status { font-size: 12px; color: #8b949e; }
  #log { display: flex; flex-direction: column; gap: 6px; }
  .entry {
    border-left: 3px solid #30363d;
    background: #161b22;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 13px;
  }
  .entry .row { display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }
  .step { color: #8b949e; font-size: 11px; min-width: 46px; }
  .ts { color: #6e7681; font-size: 11px; }
  .type { font-weight: 600; }
  .payload {
    margin-top: 4px; white-space: pre-wrap; word-break: break-all;
    color: #c9d1d9; font-size: 12px; max-height: 240px; overflow-y: auto;
  }
  .entry.tool_call { border-left-color: #58a6ff; }
  .entry.tool_result { border-left-color: #3fb950; }
  .entry.model_output { border-left-color: #a5a5ff; }
  .entry.error { border-left-color: #f85149; background: #2d1416; }
  .entry.session_start, .entry.session_end { border-left-color: #d29922; }
  .entry.warning { border-left-color: #f85149; }
  .badge {
    display: inline-block; padding: 1px 6px; border-radius: 3px;
    font-size: 11px; background: #21262d;
  }
</style>
</head>
<body>
<header>
  <span id="dot"></span>
  <h1>MAA+Agent 实时日志</h1>
  <span id="status">连接中...</span>
</header>
<div id="log"></div>
<script>
  const logEl = document.getElementById("log");
  const dot = document.getElementById("dot");
  const status = document.getElementById("status");

  function fmtTime(ts) {
    try { return new Date(ts).toLocaleTimeString("zh-CN", { hour12: false }); }
    catch (e) { return ts; }
  }

  function render(evt) {
    const payloadStr = JSON.stringify(evt.payload ?? {}, null, 2);
    const isWarning = payloadStr.includes("sensitive_warning") || payloadStr.includes("CoordinateRejected");
    const div = document.createElement("div");
    div.className = "entry " + (isWarning ? "warning" : (evt.event || ""));
    div.innerHTML = `
      <div class="row">
        <span class="step">${evt.step != null ? "步骤 " + evt.step : ""}</span>
        <span class="type badge">${evt.event}</span>
        <span class="ts">${fmtTime(evt.ts)}</span>
      </div>
      <div class="payload"></div>
    `;
    div.querySelector(".payload").textContent = payloadStr;
    logEl.appendChild(div);
    window.scrollTo(0, document.body.scrollHeight);
  }

  function connect() {
    const es = new EventSource("/events");
    es.onopen = () => { dot.className = "live"; status.textContent = "已连接"; };
    es.onerror = () => { dot.className = "dead"; status.textContent = "连接断开，重连中..."; };
    es.onmessage = (e) => {
      try { render(JSON.parse(e.data)); } catch (err) { /* ignore malformed */ }
    };
  }
  connect();
</script>
</body>
</html>
"""


class LogBroadcaster:
    """常驻本地网页，实时展示Agent探索过程中的日志（LLM调用/工具调用/敏感拦截等）"""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self._loop = asyncio.new_event_loop()
        self._subscribers: set[asyncio.Queue] = set()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        app = Starlette(routes=[
            Route("/", self._index),
            Route("/events", self._events),
        ])
        config = uvicorn.Config(app, host=self.host, port=self.port, loop="none", log_level="warning")
        self._server = uvicorn.Server(config)
        self._loop.run_until_complete(self._server.serve())

    async def _index(self, request: Request) -> HTMLResponse:
        return HTMLResponse(_DASHBOARD_HTML)

    async def _events(self, request: Request) -> StreamingResponse:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(queue)

        async def stream():
            try:
                while True:
                    payload = await queue.get()
                    yield f"data: {payload}\n\n"
            finally:
                self._subscribers.discard(queue)

        return StreamingResponse(stream(), media_type="text/event-stream")

    def publish(self, event: dict[str, Any]) -> None:
        """线程安全：可从任意线程调用，把事件塞进每个已连接前端的队列"""
        event.setdefault("ts", datetime.now().isoformat())
        payload = json.dumps(event, ensure_ascii=False, default=str)

        def _dispatch() -> None:
            for queue in list(self._subscribers):
                queue.put_nowait(payload)

        if self._loop.is_running():
            self._loop.call_soon_threadsafe(_dispatch)

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)


def attach_broadcaster(trace_logger: Any, broadcaster: LogBroadcaster) -> None:
    """给已经构造好的TraceLogger实例包一层：log_event照常落盘，同时广播到前端"""
    original_log_event: Callable[..., None] = trace_logger.log_event

    def patched_log_event(event: str, payload: dict[str, Any], step: int | None = None) -> None:
        original_log_event(event, payload, step)
        broadcaster.publish(
            {
                "session_id": trace_logger.session_id,
                "event": event,
                "step": step,
                "payload": payload,
            }
        )

    trace_logger.log_event = patched_log_event
