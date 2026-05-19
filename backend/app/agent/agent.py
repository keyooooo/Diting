import asyncio
import json
import os
import threading
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage

from app.agent.storage import storage
from app.agent.tools import get_current_weather, create_search_knowledge_tool, get_last_rag_context, reset_tool_call_guards, set_rag_step_queue

load_dotenv()
API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")

_model = init_chat_model(
    model=MODEL,
    model_provider="openai",
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.3,
    stream_usage=True,
)

_agents: dict[str, Any] = {}
_agents_lock = threading.Lock()


def get_agent_for_collection(collection_name: str, enable_rag: bool = False):
    """按 collection_name + rag 开关获取或创建 Agent 实例。"""
    cache_key = f"{collection_name}:rag={enable_rag}"
    if cache_key not in _agents:
        with _agents_lock:
            if cache_key not in _agents:
                tools = [get_current_weather]
                if enable_rag:
                    tools.append(create_search_knowledge_tool(collection_name))
                    system_prompt = (
                        "You are 菲比, a warm and enthusiastic cat bot powered by Ark API. "
                        "You have two tools:\n"
                        "- get_current_weather: for weather queries.\n"
                        "- search_knowledge_base: for searching the user's uploaded documents.\n\n"
                        "CRITICAL RULES:\n"
                        "1. Always call search_knowledge_base first for ANY question.\n"
                        "2. If the search returns documents: answer based on them. Do NOT say '文档中未找到'.\n"
                        "3. Only if the search returns zero documents or 'No relevant documents found', reply: '文档中未找到相关内容，以下是我基于自身知识的回答：' then answer from your own knowledge.\n"
                        "4. At most one search_knowledge_base call per turn.\n"
                        "5. After receiving search results, give the final answer immediately — no more tools."
                    )
                else:
                    system_prompt = (
                        "You are 菲比, a warm and enthusiastic cat bot powered by Ark API. "
                        "Chat naturally with users — answer questions, tell jokes, write code, give opinions. "
                        "You have one tool:\n"
                        "- get_current_weather: for weather queries.\n"
                        "You do NOT have access to document search right now. "
                        "Answer all non-weather questions using your own knowledge and personality."
                    )
                _agents[cache_key] = create_agent(
                    model=_model,
                    tools=tools,
                    system_prompt=system_prompt,
                )
    return _agents[cache_key]

def summarize_old_messages(model, messages: list) -> str:
    """将旧消息总结为摘要"""
    # 提取旧对话
    old_conversation = "\n".join([
        f"{'用户' if msg.type == 'human' else 'AI'}: {msg.content}"
        for msg in messages
    ])

    # 生成摘要
    summary_prompt = f"""请总结以下对话的关键信息：

{old_conversation}
总结（包含用户信息、重要事实、待办事项）："""

    summary = _model.invoke(summary_prompt).content
    return summary


def chat_with_agent(user_text: str, user_id: str = "default_user", session_id: str = "default_session", *, collection_name: str = "default", enable_rag: bool = False):
    """使用 Agent 处理用户消息并返回响应"""
    messages = storage.load(user_id, session_id)

    # 清理可能残留的 RAG 上下文，避免跨请求污染
    get_last_rag_context(clear=True)
    reset_tool_call_guards()

    if len(messages) > 50:
        summary = summarize_old_messages(_model, messages[:40])
        # 保留最近( messages - 40 ) + 1 条。
        messages = [
            SystemMessage(content=f"之前的对话摘要：\n{summary}")
        ] + messages[40:]

    # ( messages - 40 ) + 1 条 再加一条用户的输入
    agent = get_agent_for_collection(collection_name, enable_rag=enable_rag)
    messages.append(HumanMessage(content=user_text))
    result = agent.invoke(
        {"messages": messages},
        config={"recursion_limit": 8},
    )

    # 五种兜底路径，覆盖了 LangChain 不同版本 agent.invoke() 可能返回的各种格式。这种写法看起来啰嗦，但生产环境中：LangChain 小版本升级可能改变返回结构，多路径兜底保证不崩。
    response_content = ""
    if isinstance(result, dict):
        if "output" in result:
            response_content = result["output"]
        elif "messages" in result and result["messages"]:
            msg = result["messages"][-1]
            response_content = getattr(msg, "content", str(msg))
        else:
            response_content = str(result)
    elif hasattr(result, "content"):
        response_content = result.content
    else:
        response_content = str(result)
    
    # ( messages - 40 ) + 1 条 再加一条用户的输入 再加一条系统的输出
    messages.append(AIMessage(content=response_content))

    rag_context = get_last_rag_context(clear=True) # 取 tool 执行时存的 trace
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    # 存储：保存对话（只有最后一条有rag_trace ）
    extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(user_id, session_id, messages, extra_message_data=extra_message_data)

    return {
        "response": response_content,
        "rag_trace": rag_trace,
    } #前端拿到 response 展示答案，rag_trace 渲染检索诊断面板。


async def chat_with_agent_stream(user_text: str, user_id: str = "default_user", session_id: str = "default_session", *, collection_name: str = "default", enable_rag: bool = False):
    """使用 Agent 处理用户消息并流式返回响应。

    架构：使用统一输出队列 + 后台任务，确保 RAG 检索步骤在工具执行期间实时推送，
    而非等待工具完成后才显示。
    """
    messages = storage.load(user_id, session_id)

    # 清理可能残留的 RAG 上下文
    get_last_rag_context(clear=True)
    reset_tool_call_guards()

    # 统一输出队列：所有事件（content / rag_step）都汇入这里
    output_queue = asyncio.Queue()

    class _RagStepProxy:
        """代理对象：将 emit_rag_step 的原始 step dict 包装后放入统一输出队列。"""
        def put_nowait(self, step):
            output_queue.put_nowait({"type": "rag_step", "step": step})

    set_rag_step_queue(_RagStepProxy())

    if len(messages) > 50:
        summary = summarize_old_messages(_model, messages[:40])
        messages = [
            SystemMessage(content=f"之前的对话摘要：\n{summary}")
        ] + messages[40:]

    messages.append(HumanMessage(content=user_text))

    agent = get_agent_for_collection(collection_name, enable_rag=enable_rag)
    full_response = ""

    async def _agent_worker():
        """后台任务：运行 agent 并将内容 chunk 推入输出队列。"""
        nonlocal full_response
        try:
            async for msg, metadata in agent.astream(
                {"messages": messages},
                stream_mode="messages",
                config={"recursion_limit": 8},
            ):
                if not isinstance(msg, AIMessageChunk):
                    continue
                if getattr(msg, "tool_call_chunks", None):
                    continue

                content = ""
                if isinstance(msg.content, str):
                    content = msg.content
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if isinstance(block, str):
                            content += block
                        elif isinstance(block, dict) and block.get("type") == "text":
                            content += block.get("text", "")

                if content:
                    full_response += content
                    await output_queue.put({"type": "content", "content": content})
        except Exception as e:
            await output_queue.put({"type": "error", "content": str(e)})
        finally:
            # 哨兵：通知主循环 agent 已完成
            await output_queue.put(None)

    # 启动后台任务
    agent_task = asyncio.create_task(_agent_worker())

    try:
        # 主循环：持续从统一队列取事件并 yield SSE
        # RAG 步骤在工具执行期间通过 call_soon_threadsafe 实时入队，不需要等 agent 产出 chunk
        while True:
            event = await output_queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"
    except GeneratorExit:
        # 客户端断开连接（AbortController）时，FastAPI 会向此生成器抛出 GeneratorExit
        # 我们必须在此处取消后台任务
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass  # 任务已成功取消
        raise  # 重新抛出 GeneratorExit 以便 FastAPI 正确处理关闭
    finally:
        # 正常结束或异常退出时清理
        set_rag_step_queue(None)
        if not agent_task.done():
             agent_task.cancel()

    # 获取 RAG trace
    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    # 发送 trace 信息
    if rag_trace:
        yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace})}\n\n"

    # 发送结束信号
    yield "data: [DONE]\n\n"

    # 保存对话
    messages.append(AIMessage(content=full_response))
    extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
    storage.save(user_id, session_id, messages, extra_message_data=extra_message_data)
