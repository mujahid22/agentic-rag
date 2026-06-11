import os
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, TypedDict, Generator

from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage, BaseMessage, AIMessageChunk
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from .prompts import SYSTEM_PROMPT
from .tools import ALL_TOOLS
from .tool_selector import select_tools
from src.utils.query_logger import start_query, end_query, get_current_query_id, set_current_query_id

MAX_ITERATIONS = 6


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    selected_tool_names: list[str]
    iterations: int


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API_KEY"],
        model=os.environ.get("GENERATION_MODEL", "meta/llama-3.3-70b-instruct"),
        temperature=0,
    )


def _pre_select_tools_node(state: AgentState) -> dict:
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    query = last_human.content if last_human else ""
    tool_names = select_tools(query, top_k=2)
    return {"selected_tool_names": tool_names, "iterations": 0}


def _call_model_node(state: AgentState) -> dict:
    # Which tools have already been called in this session
    already_called = {
        call["name"]
        for msg in state["messages"]
        if hasattr(msg, "tool_calls")
        for call in msg.tool_calls
    }

    # Only offer tools that haven't been called yet — prevents same-tool loops
    remaining_tools = [
        ALL_TOOLS[n]
        for n in state["selected_tool_names"]
        if n in ALL_TOOLS and n not in already_called
    ]

    llm = _get_llm()

    if state["iterations"] >= MAX_ITERATIONS or not remaining_tools:
        # No tools left (or safety limit hit) — force a final text answer
        response = llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
        )
    else:
        response = llm.bind_tools(remaining_tools).invoke(
            [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
        )

    return {"messages": [response], "iterations": state["iterations"] + 1}


def _execute_tools_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_results = []

    for tool_call in last_message.tool_calls:
        tool_fn = ALL_TOOLS.get(tool_call["name"])
        if tool_fn:
            result = tool_fn.invoke(tool_call["args"])
        else:
            result = f"Tool '{tool_call['name']}' not found."

        tool_results.append(
            ToolMessage(content=result, tool_call_id=tool_call["id"])
        )

    return {"messages": tool_results}


def _should_continue(state: AgentState) -> str:
    if state["iterations"] >= MAX_ITERATIONS:
        return "end"
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "continue"
    return "end"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("pre_select_tools", _pre_select_tools_node)
    graph.add_node("call_model", _call_model_node)
    graph.add_node("execute_tools", _execute_tools_node)

    graph.add_edge(START, "pre_select_tools")
    graph.add_edge("pre_select_tools", "call_model")
    graph.add_conditional_edges(
        "call_model",
        _should_continue,
        {"continue": "execute_tools", "end": END},
    )
    graph.add_edge("execute_tools", "call_model")

    return graph.compile()


# Singleton — compiled once per process
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_graph()
    return _agent


def run_query(query: str) -> str:
    agent = get_agent()
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    last = result["messages"][-1]
    return last.content


def stream_query(query: str) -> Generator[str, None, None]:
    """
    Fast path: select tools with keywords, run them in parallel, call LLM once.
    Cuts LLM calls from 2-3 sequential to 1, targeting <30s response time.
    """
    start_query(query)

    tool_names = select_tools(query, top_k=2)
    qid = get_current_query_id()

    def _run_tool(name: str) -> str:
        set_current_query_id(qid)  # propagate query ID into each worker thread
        try:
            return ALL_TOOLS[name].invoke({"query": query})
        except Exception as exc:
            return f"[Tool {name} unavailable: {exc}]"

    active = [n for n in tool_names if n in ALL_TOOLS]
    with ThreadPoolExecutor(max_workers=max(len(active), 1)) as pool:
        tool_outputs = list(pool.map(_run_tool, active))

    context = "\n\n---\n\n".join(tool_outputs)

    llm = _get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"{query}\n\n[Retrieved Documentation]\n{context}"),
    ]

    usage = None
    is_estimate = False
    response_text = ""
    try:
        for chunk in llm.stream(messages, stream_usage=True):
            if chunk.content:
                response_text += chunk.content
                yield chunk.content
            if chunk.usage_metadata:
                usage = chunk.usage_metadata
    finally:
        if usage is None:
            is_estimate = True
            usage = _estimate_tokens(SYSTEM_PROMPT + messages[-1].content, response_text)
        end_query(usage, is_estimate=is_estimate)


def _estimate_tokens(prompt_text: str, response_text: str) -> dict:
    """Rough fallback when the API doesn't return usage_metadata: ~4 chars/token."""
    input_tokens = max(1, len(prompt_text) // 4)
    output_tokens = max(1, len(response_text) // 4)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def stream_with_status(query: str) -> Generator[tuple[str, str], None, None]:
    """
    Yields (event_type, content) tuples for real-time UI updates.
      ("status", text) — a behind-the-scenes step to display in the status box
      ("token",  text) — a token of the final answer to stream into the response area
    """
    agent = get_agent()
    synthesizing_emitted = False

    for mode, data in agent.stream(
        {"messages": [HumanMessage(content=query)]},
        stream_mode=["updates", "messages"],
    ):
        if mode == "updates":
            node_name = next(iter(data))
            node_data = data[node_name]

            if node_name == "pre_select_tools":
                tool_names = node_data.get("selected_tool_names", [])
                labels = [t.replace("tool_", "").replace("_", " ").title() for t in tool_names]
                yield ("status", f"Tools selected: {', '.join(labels)}")

            elif node_name == "call_model":
                messages = node_data.get("messages", [])
                for msg in messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            label = tc["name"].replace("tool_", "").replace("_", " ").title()
                            yield ("status", f"Calling: {label}")

            elif node_name == "execute_tools":
                messages = node_data.get("messages", [])
                for msg in messages:
                    content = getattr(msg, "content", "")
                    if "[Retrieval:" in content:
                        line = content.split("\n")[0].strip("[]")
                        if "direct" in line:
                            yield ("status", "Retrieved chunks — high-confidence match, re-rank skipped")
                        elif "bm25" in line.lower():
                            yield ("status", "Retrieved chunks from BM25 index — re-ranked with Pinecone")
                        else:
                            yield ("status", "Retrieved chunks from Vector DB — re-ranked with Pinecone")
                    else:
                        yield ("status", "Retrieved chunks from knowledge base")

        elif mode == "messages":
            chunk, _ = data
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                if not synthesizing_emitted:
                    yield ("status", "Synthesizing answer...")
                    synthesizing_emitted = True
                yield ("token", chunk.content)
