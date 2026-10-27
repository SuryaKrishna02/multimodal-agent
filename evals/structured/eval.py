"""Structured output eval: how often is the JSON valid against the schema?

Compares two ways of asking:
    constrained   response_format=json_schema (llama.cpp turns the schema into a grammar)
    prompt_only   the schema is only described in the prompt

Gold set: data/gold/structured/items.jsonl, one JSON object per line:
    {"id": "s01", "schema": "action_items", "text": "Call mum and pay rent by Friday."}
    {"id": "s02", "schema": "screen_help", "image": "s02.png"}
    {"id": "s03", "schema": "action_items", "audio": "s03.wav"}
"schema" is one of the task schemas below; "text", "image", "audio" are optional inputs
(media paths are relative to the gold folder).
"""
import json
from pathlib import Path

import jsonschema

from agent import brain, tasks
from agent.senses import audio
from evals.common import NoGold

LEVEL, USES_LLM = "component", True
SCHEMAS = {"action_items": (tasks.ACTION_SCHEMA, tasks.ACTION_PROMPT),
           "screen_help": (tasks.SCREEN_SCHEMA, tasks.SCREEN_PROMPT)}


def load_gold(gold_dir: Path):
    f = gold_dir / "items.jsonl"
    if not f.exists():
        raise NoGold(gold_dir)
    items = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return [{**it, "_dir": gold_dir} for it in items]


def _parts(item, instruction):
    parts = []
    if item.get("image"):
        parts.append(brain.part("image", item["_dir"] / item["image"]))
    if item.get("audio"):
        parts.append(brain.part("audio", audio.load_wav(item["_dir"] / item["audio"])))
    parts.append(brain.part("text", (item.get("text", "") + "\n\n" + instruction).strip()))
    return parts


def _valid(content: str, schema: dict) -> bool:
    try:
        jsonschema.validate(json.loads(content), schema)
        return True
    except (json.JSONDecodeError, jsonschema.ValidationError):
        return False


def run_item(item):
    schema, instruction = SCHEMAS[item["schema"]]
    constrained, _ = brain.ask_json(_parts(item, instruction), schema, retries=0)
    loose_prompt = instruction + "\nReply with only a JSON object matching this schema:\n" + json.dumps(schema)
    msgs = [{"role": "user", "content": _parts(item, loose_prompt)}]
    raw = brain.send(msgs)["choices"][0]["message"].get("content") or ""
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return {"id": item["id"], "constrained_valid": constrained is not None,
            "prompt_only_valid": _valid(raw, schema), "prompt_only_output": raw[:300]}


def score(records):
    n = max(len(records), 1)
    return {"valid_rate": round(sum(r["constrained_valid"] for r in records) / n, 3),
            "prompt_only_valid_rate": round(sum(r["prompt_only_valid"] for r in records) / n, 3)}
