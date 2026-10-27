"""Serving: start/stop llama-server and talk to it over its OpenAI-compatible API.

Two servers run side by side:
  llm   - Gemma 4 E4B or E2B (text + image + audio in, text out)   port 9931
  embed - EmbeddingGemma-2 (text + image + audio -> 768-d vector)  port 9932
"""
import json
import subprocess
import time
from pathlib import Path

import httpx
import psutil

from agent import config, events


class LlamaServer:
    def __init__(self, name: str, model: str, mmproj: str, port: int, extra: list[str]):
        self.name, self.port = name, port
        self.url = f"http://127.0.0.1:{port}"
        self.args = [
            str(config.LLAMA_BIN / "llama-server"),
            "-m", str(config.MODELS / model),
            "--mmproj", str(config.MODELS / mmproj),
            "--host", "127.0.0.1", "--port", str(port),   # localhost only: nothing leaves the Mac
            "--no-webui", *extra,
        ]
        self.proc: subprocess.Popen | None = None

    def healthy(self) -> bool:
        try:
            return httpx.get(self.url + "/health", timeout=1).status_code == 200
        except httpx.HTTPError:
            return False

    def start(self, timeout_s: float = 120) -> None:
        running = self.running_model()
        if running == Path(self.args[self.args.index("-m") + 1]).name and self.healthy():
            return                               # already running (e.g. started by the CLI)
        if running:                              # a different model holds the port: swap it out
            self.stop()
        config.RUNTIME.mkdir(parents=True, exist_ok=True)
        log = open(config.RUNTIME / f"llama-{self.name}.log", "w")
        events.emit("serving.start", server=self.name, cmd=" ".join(self.args))
        t0 = time.time()
        self.proc = subprocess.Popen(self.args, stdout=log, stderr=subprocess.STDOUT)
        while time.time() - t0 < timeout_s:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server '{self.name}' exited; see {log.name}")
            if self.healthy():
                events.emit("serving.ready", server=self.name, load_s=round(time.time() - t0, 1))
                return
            time.sleep(0.5)
        raise TimeoutError(f"llama-server '{self.name}' did not become healthy")

    def stop(self) -> None:
        procs = self._processes()
        for p in procs:
            p.terminate()
        psutil.wait_procs(procs, timeout=10)    # free the port before anyone restarts it
        self.proc = None
        events.emit("serving.stop", server=self.name)

    def running_model(self) -> str | None:
        """File name of the model the server on this port is serving, if any."""
        for p in self._processes():
            cmd = p.cmdline()
            return Path(cmd[cmd.index("-m") + 1]).name
        return None

    def rss_gb(self) -> float:
        """Resident memory of this server. Metal buffers live in unified memory and count here."""
        return sum(p.memory_info().rss for p in self._processes()) / 1e9

    def _processes(self) -> list[psutil.Process]:
        found = []
        for p in psutil.process_iter(["name", "cmdline"]):
            cmd = p.info["cmdline"] or []
            if p.info["name"] == "llama-server" and str(self.port) in cmd:
                found.append(p)
        return found


def llm_server(tier: str | None = None) -> LlamaServer:
    t = config.TIERS[tier or config.TIER]
    return LlamaServer("llm", t["model"], t["mmproj"], config.LLM_PORT,
                       ["-c", str(config.LLM_CTX), "-np", "1", "--metrics"])


def embed_server() -> LlamaServer:
    return LlamaServer("embed", config.EMBED["model"], config.EMBED["mmproj"], config.EMBED_PORT,
                       ["--embeddings", "-c", "8192", "-np", "1"])


# --- Client ------------------------------------------------------------------

def chat(messages: list[dict], **params) -> dict:
    """One /v1/chat/completions call. Returns the raw JSON response."""
    body = {
        "messages": messages,
        "max_tokens": config.MAX_OUTPUT_TOKENS,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": config.THINKING},
        **params,
    }
    r = httpx.post(f"http://127.0.0.1:{config.LLM_PORT}/v1/chat/completions", json=body, timeout=600)
    r.raise_for_status()
    return r.json()


def stream_timing(messages: list[dict], max_tokens: int = 128) -> dict:
    """Stream a reply to measure time to first token and generation speed.

    cache_prompt=False: otherwise a repeated prompt is served from llama.cpp's prompt cache
    and time to first token looks unrealistically fast.
    """
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": config.THINKING}}
    t0 = time.time()
    ttft, n_tokens, timings = None, 0, {}
    url = f"http://127.0.0.1:{config.LLM_PORT}/v1/chat/completions"
    with httpx.stream("POST", url, json=body, timeout=600) as r:
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            delta = chunk["choices"][0]["delta"] if chunk.get("choices") else {}
            if delta.get("content") and ttft is None:
                ttft = time.time() - t0
            n_tokens += 1 if delta.get("content") else 0
            timings = chunk.get("timings", timings)
    return {"ttft_s": ttft, "total_s": time.time() - t0,
            "gen_tok_s": timings.get("predicted_per_second"),
            "prompt_tok_s": timings.get("prompt_per_second"),
            "prompt_tokens": timings.get("prompt_n")}


def embed(inputs: list) -> list[list[float]]:
    """Embed strings, or {"content": [...]} objects holding an image or audio part."""
    r = httpx.post(f"http://127.0.0.1:{config.EMBED_PORT}/v1/embeddings", json={"input": inputs},
                   timeout=300)
    r.raise_for_status()
    return [d["embedding"] for d in r.json()["data"]]
