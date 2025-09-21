# slm_agents.py
# Tiny multi-agent LangGraph app for EDGE devices
# - Router: strictly local Ollama SLM (no keyword fallback except ultra-narrow alert safety).
# - Tools: math, joke, echo, alert, sysinfo (CPU/RAM/Disk), readme (local doc Q&A/summarize).
# - Memory: LangGraph MemorySaver + thread_id.
# - Every user-visible reply is LLM-generated (no fixed hard-coded final texts).

from typing import TypedDict, List, Literal, Optional, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage

import os
import psutil
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

# -------------
# Ollama LLM
# -------------
try:
    from langchain_ollama import ChatOllama
except Exception:
    ChatOllama = None  # type: ignore

# If True, we apply a VERY NARROW safeguard: if "alert" appears in user text, route=alert.
FORCE_ALERT_ON_KEYWORD = True

# -------------
# Graph state
# -------------
class GraphState(TypedDict, total=False):
    messages: List[BaseMessage]
    route: Optional[Literal["math", "joke", "alert", "echo", "sysinfo", "readme", "done"]]
    alerts: List[Dict[str, Any]]

# -------------
# Helpers
# -------------
def _to_text(msg: BaseMessage) -> str:
    c = getattr(msg, "content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for seg in c:
            if isinstance(seg, str):
                parts.append(seg)
            elif isinstance(seg, dict) and "text" in seg:
                parts.append(str(seg["text"]))
            else:
                parts.append(str(seg))
        return " ".join(p for p in parts if p)
    return str(c)

def _last_user_text(messages: List[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return _to_text(m)
    return _to_text(messages[-1]) if messages else ""

def _llm(base_url: str, model_name: str, temp: float = 0.3):
    return ChatOllama(base_url=base_url, model=model_name, temperature=temp)

def _llm_say(chat: Any, system_text: str, user_text: str) -> str:
    out = chat.invoke([SystemMessage(content=system_text),
                       HumanMessage(content=user_text)])
    return (getattr(out, "content", "") or "").strip()

def check_memory(limit_gb: float = 4.0):
    proc = psutil.Process(os.getpid())
    mem_gb = proc.memory_info().rss / (1024 ** 3)
    print(f"[mem] Current usage: {mem_gb:.2f} GB")
    if mem_gb > limit_gb:
        raise MemoryError(f"Memory cap exceeded: {mem_gb:.2f} GB > {limit_gb} GB")

# -------------
# Router (LLM only, with narrow safety override for 'alert')
# -------------
ROUTER_SYS = """You are a strict intent router. Look ONLY at the user's last message and return EXACTLY one label:

- math    : math expressions/calculations ("2+3*5", "compute 10/2", etc.)
- joke    : joke requests ("tell me a joke", "another joke", "funny", etc.)
- alert   : wants to create/raise/send an alert or notification (words like alert/raise/make/send warning/critical)
- echo    : general chat, paraphrasing, confirmations (no other tool matches)
- sysinfo : requests about system/device status, cpu, memory, disk, uptime, battery, performance, temperature
- readme  : asks to summarize or answer questions from local docs (e.g., 'summarize readme', 'explain installation from local file')
- done    : user says quit/exit/bye

Rules:
- Output MUST be one of: math | joke | alert | echo | sysinfo | readme | done
- Prefer 'alert' if message includes 'alert' with create/raise/send/make.
- Do not add punctuation or explanations—return only the one word.

Examples:
User: "Create alert \"Got new email\""
You: alert

User: "system status please"
You: sysinfo

User: "summarize readme about install"
You: readme

User: "2+2*5"
You: math

User: "tell me a joke"
You: joke

User: "quit"
You: done

User: "hello"
You: echo
"""

def build_llm_router_or_fail(model_name: Optional[str], base_url: str):
    if ChatOllama is None:
        raise RuntimeError("langchain-ollama not installed. Run: pip install -U langchain-ollama")
    if not model_name:
        raise RuntimeError("No model provided. Pass an Ollama tag, e.g. --model phi3:mini")

    chat = _llm(base_url, model_name, temp=0.0)

    def llm_router(state: GraphState) -> GraphState:
        user_text = _last_user_text(state["messages"]).strip()

        # ultra-narrow keyword override for safety
        if FORCE_ALERT_ON_KEYWORD and re.search(r"\balert\b", user_text, flags=re.I):
            label = "alert"
        else:
            resp = chat.invoke([SystemMessage(content=ROUTER_SYS),
                                HumanMessage(content=user_text)])
            label = (getattr(resp, "content", "") or "").strip().lower()
            if label not in {"math", "joke", "alert", "echo", "sysinfo", "readme", "done"}:
                label = "echo"

        state["messages"].append(AIMessage(content=f"[router:slm] {label}"))
        state["route"] = label  # type: ignore
        return state

    return llm_router

# -------------
# Common wrapper: push tool result through LLM for user-facing text
# -------------
def llm_wrap_answer(base_url: str, model_name: str, tool_name: str, tool_payload: Dict[str, Any], user_text: str) -> str:
    chat = _llm(base_url, model_name, temp=0.4)
    sys_txt = (
        "You are an assistant on a tiny edge device. "
        "You MUST write a clear, concise answer for the user based on the given tool output.\n"
        f"- Tool name: {tool_name}\n"
        "- Do not invent facts. Use only what is in the tool output and the user request.\n"
        "- If there is a computed result, show it first. Then briefly explain in plain English.\n"
        "- Keep it 3–8 short lines unless the user explicitly asked for long output.\n"
    )
    usr_txt = f"User said: {user_text}\n\nTool output (JSON-like):\n{tool_payload}"
    return _llm_say(chat, sys_txt, usr_txt)

# -------------
# Workers / Tools
# -------------
def make_math_agent(model_name: str, base_url: str):
    def _agent(state: GraphState) -> GraphState:
        check_memory(limit_gb=3.0)
        user_text = _last_user_text(state["messages"]).strip()
        # compute safely-ish (for demo). For production use sympy or a safe evaluator.
        try:
            result = eval(user_text, {}, {})
            payload = {"ok": True, "expression": user_text, "result": result}
        except Exception as e:
            payload = {"ok": False, "expression": user_text, "error": str(e)}
        reply = llm_wrap_answer(base_url, model_name, "math", payload, user_text)
        state["messages"].append(AIMessage(content=reply))
        state["route"] = "done"  # type: ignore
        return state
    return _agent

def make_joke_agent(model_name: str, base_url: str):
    def _agent(state: GraphState) -> GraphState:
        check_memory(limit_gb=3.0)
        user_text = _last_user_text(state["messages"])
        chat = _llm(base_url, model_name, temp=0.8)
        prompt = [
            SystemMessage(content="You are a witty assistant. Tell concise, clean programming jokes (1–2 lines). Avoid repetition in a session."),
            HumanMessage(content=f'User asked: "{user_text}". Give one new programming joke.')
        ]
        try:
            resp = chat.invoke(prompt)
            joke_text = (getattr(resp, "content", "") or "").strip()
            payload = {"ok": True, "joke": joke_text}
        except Exception as e:
            payload = {"ok": False, "error": str(e)}
        # Even the joke is passed again for uniformity
        reply = llm_wrap_answer(base_url, model_name, "joke", payload, user_text)
        state["messages"].append(AIMessage(content=reply))
        state["route"] = "done"  # type: ignore
        return state
    return _agent

_ALERT_LEVELS = ["info", "warning", "error", "critical"]

def _parse_alert(text: str):
    t = text.strip()
    lvl = None
    for l in _ALERT_LEVELS:
        if re.search(rf"\b{l}\b", t, flags=re.I):
            lvl = l
            break
    if lvl is None:
        lvl = "info"

    msg = None
    m = re.split(r":", t, maxsplit=1)
    if len(m) == 2 and m[1].strip():
        msg = m[1].strip()

    if not msg:
        q = re.findall(r'"([^"]+)"', t)
        if q:
            msg = q[0].strip()

    if not msg:
        msg = re.sub(r"\balert\b", "", t, flags=re.I).strip()
        msg = re.sub(r"\braise\b|\bcreate\b|\bsend\b|\bmake\b|\bgenerate\b|\badd\b", "", msg, flags=re.I).strip()
        if not msg:
            msg = "no message provided"
    return lvl.lower(), msg

def make_alert_agent(model_name: str, base_url: str):
    def _agent(state: GraphState) -> GraphState:
        check_memory(limit_gb=3.0)
        user_text = _last_user_text(state["messages"])
        lvl, msg = _parse_alert(user_text)
        alert_id = f"alrt-{uuid.uuid4().hex[:8]}"
        when = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        if "alerts" not in state or state["alerts"] is None:
            state["alerts"] = []
        item = {"id": alert_id, "level": lvl, "message": msg, "created_at": when}
        state["alerts"].append(item)
        payload = {"ok": True, "created": item, "total_alerts_in_session": len(state["alerts"])}
        reply = llm_wrap_answer(base_url, model_name, "alert", payload, user_text)
        state["messages"].append(AIMessage(content=reply))
        state["route"] = "done"  # type: ignore
        return state
    return _agent

def make_echo_agent(model_name: str, base_url: str):
    def _agent(state: GraphState) -> GraphState:
        check_memory(limit_gb=3.0)
        user_text = _last_user_text(state["messages"])
        payload = {"echo_of_user_text": user_text}
        reply = llm_wrap_answer(base_url, model_name, "echo", payload, user_text)
        state["messages"].append(AIMessage(content=reply))
        state["route"] = "done"  # type: ignore
        return state
    return _agent

def _collect_sysinfo() -> Dict[str, Any]:
    # keep names simple
    cpu_pct = psutil.cpu_percent(interval=0.6)
    mem = psutil.virtual_memory()

    # --- Windows-safe root selection (minimal change) ---
    # Use the current drive's anchor (e.g., 'C:\\' on Windows) or '/' on POSIX.
    root = Path(os.getcwd()).anchor or "/"
    try:
        disk = psutil.disk_usage(root)
        disk_mount = root
    except Exception:
        # Fallback: try the first mounted partition
        try:
            parts = psutil.disk_partitions(all=False)
            mount = parts[0].mountpoint if parts else root
            disk = psutil.disk_usage(mount)
            disk_mount = mount
        except Exception:
            # Final fallback: synthesize empty disk stats to avoid crash
            class _D:  # tiny struct-like holder
                total = used = free = 0
                percent = 0.0
            disk = _D()
            disk_mount = "unknown"
    # --- end minimal change ---

    info = {
        "cpu_percent": cpu_pct,
        "mem_percent": mem.percent,
        "mem_total_gb": round(mem.total / (1024**3), 2),
        "mem_available_gb": round(mem.available / (1024**3), 2),
        "disk_mount": disk_mount,
        "disk_total_gb": round(disk.total / (1024**3), 2),
        "disk_used_gb": round(disk.used / (1024**3), 2),
        "disk_free_gb": round(disk.free / (1024**3), 2),
        "disk_percent": getattr(disk, "percent", 0.0),
        "pid": os.getpid(),
    }
    try:
        boot = psutil.boot_time()
        info["boot_time_epoch"] = int(boot)
    except Exception:
        pass
    return info


def make_sysinfo_agent(model_name: str, base_url: str):
    def _agent(state: GraphState) -> GraphState:
        check_memory(limit_gb=3.0)
        user_text = _last_user_text(state["messages"])
        stats = _collect_sysinfo()
        reply = llm_wrap_answer(base_url, model_name, "sysinfo", stats, user_text)
        state["messages"].append(AIMessage(content=reply))
        state["route"] = "done"  # type: ignore
        return state
    return _agent

def _load_local_doc() -> Dict[str, Any]:
    # very simple local doc loader: pick the first .md/.txt under ./data
    base = Path("./data")
    base.mkdir(parents=True, exist_ok=True)
    files = list(base.glob("**/*.md")) + list(base.glob("**/*.txt"))
    if not files:
        return {"ok": False, "note": "No .md or .txt files under ./data"}
    f = files[0]
    try:
        text = f.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return {"ok": False, "file": str(f), "error": str(e)}
    # keep it small for tiny models
    head = text.strip()
    if len(head) > 4000:
        head = head[:4000] + "\n...[truncated]..."
    return {"ok": True, "file": str(f), "content": head}

def make_readme_agent(model_name: str, base_url: str):
    def _agent(state: GraphState) -> GraphState:
        check_memory(limit_gb=3.0)
        user_text = _last_user_text(state["messages"])
        doc = _load_local_doc()
        chat = _llm(base_url, model_name, temp=0.3)
        sys_txt = (
            "You are reading a LOCAL file on an edge device. "
            "Use ONLY the provided file content to answer the user's request. "
            "If the answer is not present, say briefly that the file does not contain it.\n"
            "- Keep the answer 5–10 lines unless the user asked for long form.\n"
        )
        usr_txt = f"User said: {user_text}\n\nLocal file payload:\n{doc}"
        out = _llm_say(chat, sys_txt, usr_txt)
        # For consistency, pass through wrapper too (not required, but uniform)
        payload = {"from_file": doc, "llm_raw": out}
        reply = llm_wrap_answer(base_url, model_name, "readme", payload, user_text)
        state["messages"].append(AIMessage(content=reply))
        state["route"] = "done"  # type: ignore
        return state
    return _agent

def final_agent(state: GraphState) -> GraphState:
    return state

# -------------
# Build graph
# -------------
def build_graph(model_name: Optional[str], base_url: str):
    graph = StateGraph(GraphState)

    llm_route_fn = build_llm_router_or_fail(model_name, base_url)
    math_fn = make_math_agent(model_name, base_url)
    joke_fn = make_joke_agent(model_name, base_url)
    alert_fn = make_alert_agent(model_name, base_url)
    echo_fn = make_echo_agent(model_name, base_url)
    sysinfo_fn = make_sysinfo_agent(model_name, base_url)
    readme_fn = make_readme_agent(model_name, base_url)

    graph.add_node("supervisor", llm_route_fn)
    graph.add_node("math", math_fn)
    graph.add_node("joke", joke_fn)
    graph.add_node("alert", alert_fn)
    graph.add_node("echo", echo_fn)
    graph.add_node("sysinfo", sysinfo_fn)
    graph.add_node("readme", readme_fn)
    graph.add_node("final", final_agent)

    graph.add_edge(START, "supervisor")

    def route_decider(state: GraphState):
        return state.get("route", "done")

    graph.add_conditional_edges(
        "supervisor",
        route_decider,
        {
            "math": "math",
            "joke": "joke",
            "alert": "alert",
            "echo": "echo",
            "sysinfo": "sysinfo",
            "readme": "readme",
            "done": "final",
        }
    )

    for n in ["math", "joke", "alert", "echo", "sysinfo", "readme"]:
        graph.add_edge(n, "final")
    graph.add_edge("final", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

# -------------
# CLI (stateful via checkpointer)
# -------------
if __name__ == "__main__":
    import argparse
    import uuid as _uuid

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Ollama model tag (e.g., phi3:mini, qwen2.5:3b-instruct)."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434",
        help="Ollama base URL (default: http://127.0.0.1:11434)"
    )
    args = parser.parse_args()

    model_tag = args.model.strip()
    base_url = args.base_url.strip()
    if model_tag.startswith("<"):
        raise SystemExit("Pass a REAL model tag, e.g. --model phi3:mini")

    print(f"[info] Using local Ollama model: {model_tag}")

    app = build_graph(model_tag, base_url)
    session_id = f"cli-{_uuid.uuid4()}"

    print("\nTiny Multi-Agent Demo (SLM router + tools + memory + edge extras)")
    print("Try:")
    print('  - 2+3*(7-2)')
    print('  - tell me a joke / tell me another joke')
    print('  - Create alert \"Got new email\"')
    print('  - raise alert warning: cpu usage high on node-3')
    print('  - system status please')
    print('  - summarize readme about installation steps  (put a .md/.txt under ./data)')
    print('  - hello there')
    print('  - quit\n')

    try:
        while True:
            user = input("you> ").strip()
            if not user:
                continue

            cfg = {"configurable": {"thread_id": session_id}}
            out = app.invoke({"messages": [HumanMessage(content=user)]}, config=cfg)
            ai_msgs = [m for m in out["messages"] if isinstance(m, AIMessage)]
            print("bot>", ai_msgs[-1].content if ai_msgs else "(no reply)")

            if user.lower() in {"quit", "exit", "bye"}:
                break

    except KeyboardInterrupt:
        print("\nbye!")
