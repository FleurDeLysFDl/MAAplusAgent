"""给hello_agents.ReActAgent打补丁：让get_raw_image真正把截图作为OpenAI原生
image_url内容块发给多模态LLM，而不是把base64当纯文本塞进对话历史。

背景：hello_agents.tools.response.ToolResponse只有text: str字段，ReActAgent的
ReAct循环（core/agent.py的_execute_tool_call）把工具结果原样塞进
{"role": "tool", "content": <str>}消息里发给OpenAI API。如果直接把整张截图的
base64编码放进text，模型收到的是一坨它看不懂的文本乱码（不是真图片），而且这段
文本会永久留在对话历史里、每一步都重新发一遍——实测300步任务里出现过854541
tokens（模型上限128000），几步之内就能把上下文打爆。

hello_agents没有给这个场景留任何干净的扩展点（messages列表是_run_impl的局部
变量，工具结果拼接逻辑内联在一个250行的私有方法里），所以这里monkeypatch
ReActAgent._run_impl。core/agent_runner.py只用同步的agent.run()，不涉及
arun()/arun_stream()，所以只需要patch这一个方法。

在原方法的基础上改了三处（其余逻辑原样照抄自hello_agents 1.0.0的
react_agent.py，库版本升级后需要重新对比同步一遍）：
1. 工具结果来自get_raw_image时，不把base64塞进tool消息，而是塞一条占位文本
   满足OpenAI"每个tool_call_id必须有对应tool角色回复"的要求，紧跟着追加一条
   真正的user角色多模态消息（image_url内容块），模型才能看懂图。
2. 每次调用LLM前清理旧的带图消息，只保留最近MAX_KEPT_IMAGES条的图片内容，
   更早的替换成占位文字，避免连续多次get_raw_image调用后再次把上下文撑爆。
3. 双模型路由：screenshot()返回的known_actions非空，说明当前界面之前来过、
   探索记忆里已经存了验证过坐标的操作（core/exploration_memory.py），不需要
   看图也不需要重新摸索坐标，下一步LLM调用就切到agent.routine_llm（便宜的纯
   文本模型，如果没配就还是用self.llm）；命中get_raw_image、或者又出现了
   没见过的新界面，就切回self.llm（多模态探索模型）。切到routine_llm调用时，
   messages里可能还留着最近几步的image_url内容块（MAX_KEPT_IMAGES淘汰窗口
   还没轮到它们），临时替换成占位文字再发送，不修改messages本体，避免不支持
   多模态的模型报错，也避免白白多花图片token。
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from hello_agents.agents.react_agent import ReActAgent
from hello_agents.core.message import Message

IMAGE_TOOL_NAME = "get_raw_image"
SCREENSHOT_TOOL_NAME = "screenshot"
MAX_KEPT_IMAGES = 2


def _is_image_message(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return (
        message.get("role") == "user"
        and isinstance(content, list)
        and any(isinstance(b, dict) and b.get("type") == "image_url" for b in content)
    )


def _evict_old_images(messages: list[dict[str, Any]]) -> None:
    """只保留最近MAX_KEPT_IMAGES条带图消息的图片内容，更早的换成占位文字"""
    image_indexes = [i for i, m in enumerate(messages) if _is_image_message(m)]
    stale_indexes = image_indexes[:-MAX_KEPT_IMAGES] if MAX_KEPT_IMAGES > 0 else image_indexes
    for i in stale_indexes:
        message = messages[i]
        text_blocks = [b for b in message["content"] if b.get("type") == "text"]
        text_blocks.append({"type": "text", "text": "[早前的截图原图已从上下文移除以节省token]"})
        message["content"] = text_blocks


def _strip_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """给routine_llm用的临时副本：image_url内容块换成占位文字，不改动原messages"""
    sanitized = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "image_url" for b in content
        ):
            new_content = [
                b if not (isinstance(b, dict) and b.get("type") == "image_url")
                else {"type": "text", "text": "[图片内容已省略：当前用轻量模型处理，不需要看图]"}
                for b in content
            ]
            message = {**message, "content": new_content}
        sanitized.append(message)
    return sanitized


def _build_image_messages(tool_call_id: str, base64_png: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": "[原图已作为图片消息附在下一条消息里，不是文字结果]",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "上面get_raw_image工具返回的截图原图如下，用于确认OCR识别不出来的"
                        "图标类按钮或空白区域的像素坐标（坐标原点在图片左上角，和screenshot()"
                        "的box坐标系一致），确认后调用click_on_image点击："
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_png}"},
                },
            ],
        },
    ]


def _patched_run_impl(self: ReActAgent, input_text: str, session_start_time, **kwargs) -> str:
    messages = self._build_messages(input_text)
    tool_schemas = self._build_tool_schemas()

    current_step = 0
    total_tokens = 0
    use_routine = False  # 一开始什么都没探索过，没有known_actions可用，从多模态起步

    if self.trace_logger:
        self.trace_logger.log_event("message_written", {"role": "user", "content": input_text})

    print(f"\n🤖 {self.name} 开始处理问题: {input_text}")

    while current_step < self.max_steps:
        current_step += 1
        print(f"\n--- 第 {current_step} 步 ---")
        self._current_step = current_step

        _evict_old_images(messages)

        routine_llm = getattr(self, "routine_llm", None)
        active_llm = routine_llm if (use_routine and routine_llm is not None) else self.llm
        call_messages = _strip_images(messages) if active_llm is routine_llm else messages
        if routine_llm is not None:
            print(f"🧭 本步使用{'routine（便宜）' if active_llm is routine_llm else 'explore（多模态）'}模型")

        try:
            response = active_llm.invoke_with_tools(
                messages=call_messages,
                tools=tool_schemas,
                tool_choice="auto",
                **kwargs,
            )
        except Exception as e:
            print(f"❌ LLM 调用失败: {e}")
            if self.trace_logger:
                self.trace_logger.log_event(
                    "error", {"error_type": "LLM_ERROR", "message": str(e)}, step=current_step
                )
            break

        response_message = response.choices[0].message

        if response.usage:
            total_tokens += response.usage.total_tokens
            self._total_tokens = total_tokens

        if self.trace_logger:
            self.trace_logger.log_event(
                "model_output",
                {
                    "content": response_message.content or "",
                    "tool_calls": len(response_message.tool_calls) if response_message.tool_calls else 0,
                    "usage": {
                        "total_tokens": response.usage.total_tokens if response.usage else 0,
                        "cost": 0.0,
                    },
                },
                step=current_step,
            )

        tool_calls = response_message.tool_calls
        if not tool_calls:
            final_answer = response_message.content or "抱歉，我无法回答这个问题。"
            print(f"💬 直接回复: {final_answer}")
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(final_answer, "assistant"))
            if self.trace_logger:
                duration = (datetime.now() - session_start_time).total_seconds()
                self.trace_logger.log_event(
                    "session_end",
                    {
                        "duration": duration,
                        "total_steps": current_step,
                        "final_answer": final_answer,
                        "status": "success",
                    },
                )
                self.trace_logger.finalize()
            return final_answer

        messages.append(
            {
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_call_id = tool_call.id

            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                print(f"❌ 工具参数解析失败: {e}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": f"错误：参数格式不正确 - {str(e)}",
                    }
                )
                continue

            if self.trace_logger:
                self.trace_logger.log_event(
                    "tool_call",
                    {"tool_name": tool_name, "tool_call_id": tool_call_id, "args": arguments},
                    step=current_step,
                )

            if tool_name in self._builtin_tools:
                result = self._handle_builtin_tool(tool_name, arguments)
                print(f"🔧 {tool_name}: {result['content']}")

                if self.trace_logger:
                    self.trace_logger.log_event(
                        "tool_result",
                        {
                            "tool_name": tool_name,
                            "tool_call_id": tool_call_id,
                            "status": "success",
                            "result": result["content"],
                        },
                        step=current_step,
                    )

                if tool_name == "Finish" and result.get("finished"):
                    final_answer = result["final_answer"]
                    print(f"🎉 最终答案: {final_answer}")
                    self.add_message(Message(input_text, "user"))
                    self.add_message(Message(final_answer, "assistant"))
                    if self.trace_logger:
                        duration = (datetime.now() - session_start_time).total_seconds()
                        self.trace_logger.log_event(
                            "session_end",
                            {
                                "duration": duration,
                                "total_steps": current_step,
                                "final_answer": final_answer,
                                "status": "success",
                            },
                        )
                        self.trace_logger.finalize()
                    return final_answer

                messages.append(
                    {"role": "tool", "tool_call_id": tool_call_id, "content": result["content"]}
                )
            else:
                print(f"🎬 调用工具: {tool_name}({arguments})")
                result = self._execute_tool_call(tool_name, arguments)

                if self.trace_logger:
                    self.trace_logger.log_event(
                        "tool_result",
                        {"tool_name": tool_name, "tool_call_id": tool_call_id, "result": result},
                        step=current_step,
                    )

                if result.startswith("❌"):
                    print(result)
                else:
                    print(f"👀 观察: {result}")

                if tool_name == IMAGE_TOOL_NAME and not result.startswith("❌"):
                    messages.extend(_build_image_messages(tool_call_id, result))
                    use_routine = False  # 主动要看图了，下一步必须回到多模态模型
                else:
                    messages.append(
                        {"role": "tool", "tool_call_id": tool_call_id, "content": result}
                    )
                    if tool_name == SCREENSHOT_TOOL_NAME and not result.startswith("❌"):
                        try:
                            use_routine = bool(json.loads(result).get("known_actions"))
                        except (json.JSONDecodeError, AttributeError):
                            use_routine = False

    print("⏰ 已达到最大步数，流程终止。")
    final_answer = "抱歉，我无法在限定步数内完成这个任务。"
    self.add_message(Message(input_text, "user"))
    self.add_message(Message(final_answer, "assistant"))
    if self.trace_logger:
        duration = (datetime.now() - session_start_time).total_seconds()
        self.trace_logger.log_event(
            "session_end",
            {
                "duration": duration,
                "total_steps": current_step,
                "final_answer": final_answer,
                "status": "timeout",
            },
        )
        self.trace_logger.finalize()
    return final_answer


def install_vision_support() -> None:
    """把get_raw_image真正接成多模态图片消息。需在构建ReActAgent前调用一次。"""
    ReActAgent._run_impl = _patched_run_impl
