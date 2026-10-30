"""The brain: turn what the agent sensed into one model request, and act on the reply.

1. assemble   - interleave text, image and audio parts into one user message
2. budget     - estimate tokens per part; drop old history / memories if over budget
3. structured - ask for JSON that matches a schema (grammar-constrained by llama.cpp)
4. tools      - let the model call a few tools; validate every call; retry when invalid
"""

import base64
import copy
import io
import json
import time
from collections.abc import Sequence

import jsonschema
from PIL import Image

from agent import config, events, serving, tools
from agent.senses import audio as audio_sense
from agent.senses import video as video_sense

SYSTEM = """You are a private assistant running fully on this Mac. You can hear short voice clips,
see screenshots and use tools. Be brief and concrete. Use a tool only when it helps.
Current time: {now}."""


def system_prompt() -> str:
    """Return the system prompt with the current time filled in."""
    # Minute resolution on purpose: llama.cpp reuses the cached prefix of the previous
    # prompt, and a timestamp with seconds would change the prefix on every request.
    return SYSTEM.format(now=time.strftime("%A %d %B %Y, %H:%M"))


# --- 1. Assemble ---------------------------------------------------------------


def part(kind: str, value) -> dict:
    """One content part.

    kind: "text" (str), "image" (PIL image or path), "audio" (16 kHz float32 array), "video" (file path).
    Keys starting with "_" are notes for the token budget; send() strips them.
    """
    if kind == "text":
        return {"type": "text", "text": value}
    if kind == "image":
        img = value if isinstance(value, Image.Image) else Image.open(value)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "PNG")
        return {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()},
            "_size": img.size,
        }
    if kind == "audio":
        data = base64.b64encode(audio_sense.to_wav_bytes(value)).decode()
        return {
            "type": "input_audio",
            "input_audio": {"data": data, "format": "wav"},
            "_seconds": round(len(value) / config.SAMPLE_RATE, 1),
        }
    if kind == "video":
        info = video_sense.probe(value)
        return {
            "type": "input_video",
            "input_video": {"data": base64.b64encode(open(value, "rb").read()).decode()},
            "_size": (info["width"], info["height"]),
            "_seconds": info["seconds"],
        }
    raise ValueError(kind)


# --- 2. Context budget ---------------------------------------------------------


def image_tokens(size: tuple[int, int]) -> int:
    """Estimate the tokens that an image costs.

    Args:
        size (tuple[int, int]): Image width and height in pixels.

    Returns:
        int: Estimated tokens, capped at `config.IMAGE_MAX_TOKENS`.
    """
    return min(config.IMAGE_MAX_TOKENS, size[0] * size[1] // config.IMAGE_PIXELS_PER_TOKEN)


def estimate_tokens(p: dict) -> int:
    """Estimate the tokens of one content part.

    Args:
        p (dict): A content part from `part()`, with its private `_size` or `_seconds` notes.

    Returns:
        int: Estimated tokens.
    """
    if p["type"] == "text":
        return len(p["text"]) // 4 + 1  # rough rule: ~4 characters per token
    if p["type"] == "image_url":
        return image_tokens(p.get("_size", (1280, 720)))
    if p["type"] == "input_video":  # one image per sampled frame
        return int(p["_seconds"] * config.VIDEO_FPS + 1) * image_tokens(p["_size"])
    return int(p.get("_seconds", 30) * config.AUDIO_TOKENS_PER_S)


def message_tokens(m: dict) -> int:
    """Estimate the tokens of one chat message.

    Args:
        m (dict): A chat message whose content is a string or a list of parts.

    Returns:
        int: Estimated tokens.
    """
    c = m.get("content") or ""
    return sum(estimate_tokens(p) for p in c) if isinstance(c, list) else len(c) // 4 + 1


def fit_budget(
    system: str, memories: list[str], history: list[dict], user: list[dict], tool_schemas: Sequence[dict] = ()
) -> tuple[list, dict]:
    """Build the message list; drop oldest history, then memories, until it fits.

    Tool declarations count too: the chat template writes them into the prompt.
    """
    budget = config.LLM_CTX - config.MAX_OUTPUT_TOKENS
    history, memories = list(history), list(memories)
    while True:
        sys_text = system + ("\n\nRelevant memories:\n- " + "\n- ".join(memories) if memories else "")
        msgs = [{"role": "system", "content": sys_text}, *history, {"role": "user", "content": user}]
        used = {
            "system+memories": len(sys_text) // 4,
            "tools": len(json.dumps(tool_schemas)) // 4,
            "history": sum(map(message_tokens, history)),
            "user": sum(map(estimate_tokens, user)),
        }
        if sum(used.values()) <= budget or (not history and not memories):
            used = {**used, "total": sum(used.values()), "budget": budget}
            events.emit("brain.budget", **used, memories=len(memories), history_turns=len(history))
            return msgs, used
        if history:
            history.pop(0)
        else:
            memories.pop()


def send(msgs: list[dict], **params) -> dict:
    """Strip private keys, log the exact request for the debug view, call the server."""
    clean = [
        {**m, "content": [{k: v for k, v in p.items() if not k.startswith("_")} for p in m["content"]]}
        if isinstance(m.get("content"), list)
        else m
        for m in msgs
    ]
    body = {"messages": clean, **params}
    config.RUNTIME.mkdir(parents=True, exist_ok=True)
    (config.RUNTIME / "last_request.json").write_text(json.dumps(body, indent=1))
    events.emit("brain.request", body=_elide_media(body))
    t0 = time.time()
    reply = serving.chat(clean, **params)
    events.emit(
        "brain.reply",
        seconds=round(time.time() - t0, 2),
        message=reply["choices"][0]["message"],
        timings=reply.get("timings"),
        usage=reply.get("usage"),
    )
    return reply


def _elide_part(p: dict) -> None:
    """Shorten the base64 blob of one content part, in place."""
    for key in ("image_url", "input_audio", "input_video"):
        if key in p:
            field = "url" if key == "image_url" else "data"
            p[key][field] = p[key][field][:48] + f"...<{len(p[key][field]):,} chars>"


def _elide_media(body: dict) -> dict:
    """Same request, with base64 blobs shortened so it fits on screen."""
    body = copy.deepcopy(body)
    for m in body["messages"]:
        for p in m["content"] if isinstance(m.get("content"), list) else []:
            _elide_part(p)
    return body


# --- 3. Structured output ------------------------------------------------------


def ask_json(user: list[dict], schema: dict, system: str | None = None, retries: int = 1) -> tuple[dict | None, dict]:
    """Ask for JSON matching `schema`. Returns (data or None, info)."""
    msgs, used = fit_budget(system or system_prompt(), [], [], user)
    fmt = {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}
    for attempt in range(retries + 1):
        content = send(msgs, response_format=fmt)["choices"][0]["message"].get("content") or ""
        try:
            data = json.loads(content)
            jsonschema.validate(data, schema)
            return data, {"attempts": attempt + 1, "tokens": used}
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            events.emit("brain.invalid_json", attempt=attempt + 1, error=str(e)[:200], content=content[:300])
    return None, {"attempts": retries + 1, "tokens": used}


# --- 4. Tool calls -------------------------------------------------------------


def _recall(query: str, use_memory: bool) -> list[str]:
    """Format the memories that match `query`, or return none when memory is off."""
    if not (use_memory and query):
        return []
    from agent.memory import episodic

    return [f"({m['when']}, {m['kind']}) {m['text']}" for m in episodic.search(query)]


def _execute_call(call: dict) -> tuple[str, dict, dict]:
    """Parse and run one tool call from the model. Returns (name, args, result)."""
    name = call["function"]["name"]
    try:
        args = json.loads(call["function"]["arguments"] or "{}")
    except json.JSONDecodeError:
        return name, {}, {"status": "invalid", "error": "arguments are not valid JSON"}
    return name, args, tools.run(name, args)


def _handle_calls(calls: list[dict], msgs: list[dict], trace: list[dict], retries_left: int) -> int:
    """Run the tool calls of one model turn and append the results to `msgs`.

    Stops at the first invalid call that uses up the retries, so no later call in the batch runs.
    Returns the retries left, which is negative when they are used up.
    """
    for c in calls:
        name, args, result = _execute_call(c)
        trace.append({"tool": name, "args": args, **result})
        if result["status"] == "invalid":
            retries_left -= 1
            if retries_left < 0:
                break
        msgs.append({"role": "tool", "tool_call_id": c.get("id", name), "content": json.dumps(result, default=str)})
    return retries_left


def respond(user: list[dict], history: list[dict] | None = None, use_memory: bool = True, max_steps: int = 4) -> dict:
    """A full turn: memories + tools. Returns {"text", "tool_calls", "tokens"}."""
    query = " ".join(p["text"] for p in user if p["type"] == "text")
    names = tools.select(query or "help", config.MAX_TOOLS_PER_REQUEST)
    msgs, used = fit_budget(system_prompt(), _recall(query, use_memory), history or [], user, tools.schemas(names))
    trace, retries_left = [], config.TOOL_RETRIES
    out = {"tool_calls": trace, "tokens": used, "tools_offered": names}
    for step in range(max_steps):
        # Known Gemma 4 quirk: with thinking off, the turn after a tool result comes back empty
        # (same as ollama/ollama#19044). Thinking on for that one step fixes it.
        params = {"chat_template_kwargs": {"enable_thinking": True}} if step > 0 else {}
        msg = send(msgs, tools=tools.schemas(names), **params)["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            return {"text": msg.get("content") or "", **out}
        msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        retries_left = _handle_calls(calls, msgs, trace, retries_left)
        if retries_left < 0:
            return {"text": "I could not form a valid tool call.", **out}
    return {"text": "Stopped after too many tool steps.", **out}
