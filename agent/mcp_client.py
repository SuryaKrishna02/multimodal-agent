"""MCP client: load a pasted `mcpServers` config, connect to each server, and offer its enabled tools to the model.

    {"mcpServers": {
        "reminders": {"command": "/path/to/python", "args": ["mcp_servers/reminders/server.py"]},
        "github":    {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"],
                      "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"}},
        "search":    {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer ${SEARCH_KEY}"}}}}

1. Labels. `command` starts a process on this machine that talks over stdin and stdout: on-device. `url` sends
   every call over the network: remote, unless the host is this machine. The label describes the transport only:
   a stdio server can still reach the internet itself (the GitHub server does).
2. Secrets. Each value in `env` and `headers` must be `${NAME}`. It is read from the environment when the client
   connects, and it is never written to disk. A literal value is refused, because a pasted key would end up in
   `config.MCP_CONFIG`.
3. Tools and tiers. The user enables a subset of each server's tools and sets the tier of each one. Every tool
   starts as `config.MCP_DEFAULT_TIER`, because a server's annotations (read only, destructive) are hints from code
   that we do not control.
4. Enabled tools join the built-in tools in one registry as `<server>__<tool>`, so `tools.check()` (schema, tier,
   guard) and `tools.select()` treat both kinds the same.

The SDK is async and the brain is not. One event loop runs in a background thread with one task per server. Each
task keeps its session open and answers calls from a queue (anyio needs one task to open and close a session).
"""

import asyncio
import concurrent.futures
import ipaddress
import json
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from urllib.parse import urlparse

from agent import config, events, tools

SERVER_NAME = re.compile(r"^[A-Za-z0-9_-]{1,32}$")  # becomes part of a tool name the model calls
ENV_REF = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
ON_DEVICE, REMOTE, LOCAL_HTTP = "on-device", "remote", "on-device (HTTP)"
SEPARATOR = "__"  # <server>__<tool>
TIERS = (tools.SAFE, tools.CONFIRM, tools.BLOCKED)


class ConfigError(ValueError):
    """The pasted config cannot be used. The message lists every problem."""


@dataclass
class Server:
    """One entry of `mcpServers`.

    Attributes:
        name (str): The key in `mcpServers`.
        label (str): ON_DEVICE, REMOTE or LOCAL_HTTP.
        command (str | None): The program that a stdio server runs.
        args (list[str]): Its arguments.
        env (dict[str, str]): Extra environment, as `${NAME}` references.
        cwd (str | None): Working folder of the process.
        url (str | None): The endpoint of an HTTP server.
        headers (dict[str, str]): HTTP headers, as `${NAME}` references.
    """

    name: str
    label: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


def label_of(url: str) -> str:
    """Label an HTTP server: on this machine or remote.

    Args:
        url (str): The endpoint.

    Returns:
        str: LOCAL_HTTP for localhost or a loopback address, else REMOTE.
    """
    host = urlparse(url).hostname or ""
    if host == "localhost":
        return LOCAL_HTTP
    try:
        return LOCAL_HTTP if ipaddress.ip_address(host).is_loopback else REMOTE
    except ValueError:
        return REMOTE


def _secret_problems(name: str, where: str, values) -> list[str]:
    """List the values that are not `${NAME}` references."""
    if not isinstance(values, dict):
        return [f"{name}: '{where}' must be an object"]
    return [
        f'{name}: {where}.{key} holds a literal value. Move it to an environment variable and write "${{{key}}}"'
        for key, value in values.items()
        if not (isinstance(value, str) and ENV_REF.match(value))
    ]


def _server(name: str, entry) -> tuple[Server | None, list[str]]:
    """Check one `mcpServers` entry and turn it into a Server."""
    if not SERVER_NAME.match(name):
        return None, [f"{name!r}: a server name may use only letters, digits, - and _ (at most 32)"]
    if not isinstance(entry, dict) or ("command" in entry) == ("url" in entry):
        return None, [f"{name}: give either 'command' (a local process) or 'url' (an HTTP server)"]
    problems = _secret_problems(name, "env", entry.get("env", {})) + _secret_problems(
        name, "headers", entry.get("headers", {})
    )
    if "command" in entry:
        server = Server(
            name, ON_DEVICE, command=str(entry["command"]), args=[str(a) for a in entry.get("args", [])],
            env=dict(entry.get("env", {})), cwd=entry.get("cwd"),
        )  # fmt: skip
    else:
        url = str(entry["url"])
        if urlparse(url).scheme not in ("http", "https"):
            problems.append(f"{name}: url must start with http:// or https://")
        server = Server(name, label_of(url), url=url, headers=dict(entry.get("headers", {})))
    return server, problems


def parse(text: str) -> dict[str, Server]:
    """Read a pasted config in the standard `mcpServers` format.

    Args:
        text (str): JSON with a top-level "mcpServers" object (a bare object of servers also works).

    Returns:
        dict[str, Server]: The servers by name.

    Raises:
        ConfigError: Invalid JSON, or any server with a problem. The message lists them all.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ConfigError(f"not valid JSON: {e}") from e
    entries = data.get("mcpServers", data) if isinstance(data, dict) else None
    if not isinstance(entries, dict) or not entries:
        raise ConfigError('expected {"mcpServers": {"<name>": {...}}}')
    servers, problems = {}, []
    for name, entry in entries.items():
        server, found = _server(name, entry)
        problems += found
        if server:
            servers[name] = server
    if problems:
        raise ConfigError("\n".join(problems))
    return servers


def resolve(refs: dict[str, str]) -> dict[str, str]:
    """Replace each `${NAME}` with the value of that environment variable.

    Args:
        refs (dict[str, str]): Values such as "${GITHUB_TOKEN}".

    Returns:
        dict[str, str]: The real values, for this connection only.

    Raises:
        ConfigError: A variable is not set.
    """
    out, missing = {}, []
    for key, ref in refs.items():
        match = ENV_REF.match(ref)
        var = match.group(1) if match else ""
        if var in os.environ:
            out[key] = os.environ[var]
        else:
            missing.append(var or key)
    if missing:
        raise ConfigError(f"set {', '.join(missing)} in the environment of the process that connects")
    return out


# --- The saved state: config, discovered tools, enabled tools, tiers -------------------------------


def load() -> dict:
    """Read the saved MCP state, or the default: the reminders server of this repo, nothing enabled.

    Returns:
        dict: `mcpServers` (as pasted), `tools` (what each server offered the last time), `enabled` (tool names per
        server) and `tiers` (tier per `<server>__<tool>`).
    """
    try:
        return json.loads(config.MCP_CONFIG.read_text())
    except (OSError, ValueError):
        return {"mcpServers": default_servers(), "tools": {}, "enabled": {}, "tiers": {}}


def default_servers() -> dict:
    """The reminders server of this repo, run by this Python.

    Returns:
        dict: One `mcpServers` entry.
    """
    script = config.ROOT / "mcp_servers" / "reminders" / "server.py"
    db = config.RUNTIME / "reminders.sqlite"
    return {config.REMINDERS_SERVER: {"command": sys.executable, "args": [str(script), "--db", str(db)]}}


def save(state: dict) -> None:
    """Write the MCP state (it holds only `${NAME}` references, never a secret).

    Args:
        state (dict): As returned by `load()`.
    """
    config.MCP_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    config.MCP_CONFIG.write_text(json.dumps(state, indent=1))


def set_config(text: str) -> dict[str, Server]:
    """Replace the server list with a pasted config. Enabled tools and tiers of servers that stay are kept.

    Args:
        text (str): The pasted JSON.

    Returns:
        dict[str, Server]: The parsed servers.
    """
    servers = parse(text)
    data = json.loads(text)
    state = load()
    state["mcpServers"] = data.get("mcpServers", data)
    for key in ("tools", "enabled"):
        state[key] = {k: v for k, v in state.get(key, {}).items() if k in servers}
    state["tiers"] = {k: v for k, v in state.get("tiers", {}).items() if k.split(SEPARATOR)[0] in servers}
    save(state)
    events.emit("mcp.config", servers={n: s.label for n, s in servers.items()})
    return servers


def servers() -> dict[str, Server]:
    """Return the saved servers.

    Returns:
        dict[str, Server]: By name.
    """
    return parse(json.dumps({"mcpServers": load()["mcpServers"]}))


def enable(server: str, names: list[str]) -> None:
    """Choose which tools of a server the model may see. The rest stay hidden.

    Args:
        server (str): Server name.
        names (list[str]): Tool names as the server lists them.
    """
    state = load()
    state.setdefault("enabled", {})[server] = sorted(set(names))
    save(state)
    events.emit("mcp.enabled", server=server, tools=sorted(set(names)))
    hub.register(server)


def set_tier(server: str, tool: str, tier: str) -> None:
    """Set the permission tier of one MCP tool.

    Args:
        server (str): Server name.
        tool (str): Tool name as the server lists it.
        tier (str): "safe", "confirm" or "blocked".

    Raises:
        ValueError: An unknown tier.
    """
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {', '.join(TIERS)}")
    state = load()
    state.setdefault("tiers", {})[f"{server}{SEPARATOR}{tool}"] = tier
    save(state)
    events.emit("mcp.tier", server=server, tool=tool, tier=tier)
    hub.register(server)


def tier_of(state: dict, server: str, tool: str) -> str:
    """Return the tier the user chose for a tool, or the default.

    Args:
        state (dict): As returned by `load()`.
        server (str): Server name.
        tool (str): Tool name.

    Returns:
        str: The tier.
    """
    return state.get("tiers", {}).get(f"{server}{SEPARATOR}{tool}", config.MCP_DEFAULT_TIER)


def _text_of(result) -> str | dict | list:
    """Turn a CallToolResult into what the model reads: the structured result if any, else the text parts."""
    if result.structured_content is not None:
        content = result.structured_content
        return content.get("result", content) if isinstance(content, dict) and len(content) == 1 else content
    return "\n".join(getattr(part, "text", f"[{getattr(part, 'type', 'content')}]") for part in result.content)


# --- Connections --------------------------------------------------------------------------------------


class Hub:
    """Keeps one open session per connected server, on one event loop in a background thread."""

    def __init__(self):
        """Start with no loop and no sessions; the loop starts on the first connection."""
        self.loop: asyncio.AbstractEventLoop | None = None
        self.queues: dict[str, asyncio.Queue] = {}
        self.lock = threading.Lock()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self.lock:
            if self.loop is None:
                self.loop = asyncio.new_event_loop()
                threading.Thread(target=self.loop.run_forever, name="mcp", daemon=True).start()
            return self.loop

    def connect(self, name: str) -> list[dict]:
        """Start (or restart) the session with one server, list its tools, and register the enabled ones.

        Args:
            name (str): Server name in the saved config.

        Returns:
            list[dict]: The tools that the server offers: name, description, input_schema, annotations.

        Raises:
            ConfigError: A secret is not set in the environment.
            TimeoutError: The server did not answer within `config.MCP_CONNECT_TIMEOUT_S`.
        """
        server = servers()[name]
        self.disconnect(name)
        loop = self._ensure_loop()
        ready: concurrent.futures.Future = concurrent.futures.Future()
        asyncio.run_coroutine_threadsafe(self._session(server, ready), loop)
        found = ready.result(timeout=config.MCP_CONNECT_TIMEOUT_S)
        state = load()
        state.setdefault("tools", {})[name] = found
        save(state)
        events.emit("mcp.connected", server=name, label=server.label, tools=[t["name"] for t in found])
        self.register(name)
        return found

    async def _session(self, server: Server, ready: concurrent.futures.Future) -> None:
        """Open the session, report its tools, then answer calls from the queue until told to stop."""
        from mcp import Client

        queue: asyncio.Queue = asyncio.Queue()
        try:
            async with _transport(server) as target, Client(target) as client:
                listing = await client.list_tools()
                self.queues[server.name] = queue
                ready.set_result([_tool_info(t) for t in listing.tools])
                await _answer(client, queue)
        except Exception as e:  # noqa: BLE001 - a failed server must not stop the others; the caller sees why
            if not ready.done():
                ready.set_exception(e)
            events.emit("mcp.error", server=server.name, error=f"{type(e).__name__}: {e}")
        finally:
            # A reconnect may already have put a new session's queue under this name: leave that one alone.
            if self.queues.get(server.name) is queue:
                del self.queues[server.name]

    def call(self, server: str, tool: str, args: dict):
        """Call one tool and wait for the result.

        Args:
            server (str): Server name.
            tool (str): Tool name as the server lists it.
            args (dict): Arguments, already checked against the schema by `tools.check()`.

        Returns:
            str | dict | list: What the model reads.

        Raises:
            RuntimeError: The server is not connected, or the tool reported an error.
        """
        queue = self.queues.get(server)
        if queue is None or self.loop is None:
            raise RuntimeError(f"MCP server '{server}' is not connected")
        reply: concurrent.futures.Future = concurrent.futures.Future()
        self.loop.call_soon_threadsafe(queue.put_nowait, (tool, args, reply))
        result = reply.result(timeout=config.MCP_CALL_TIMEOUT_S)
        if result.is_error:
            raise RuntimeError(str(_text_of(result)))
        return _text_of(result)

    def disconnect(self, name: str) -> None:
        """Close the session with a server and remove its tools from the registry.

        Args:
            name (str): Server name.
        """
        queue = self.queues.get(name)
        if queue is not None and self.loop is not None:
            self.loop.call_soon_threadsafe(queue.put_nowait, None)
        _unregister(name)

    def connected(self) -> list[str]:
        """List the servers with an open session.

        Returns:
            list[str]: Server names.
        """
        return sorted(self.queues)

    def register(self, name: str) -> None:
        """Put the enabled tools of a connected server into the tool registry, with their tiers.

        Args:
            name (str): Server name.
        """
        _unregister(name)
        if name not in self.queues:
            return
        state = load()
        label = servers()[name].label
        enabled = set(state.get("enabled", {}).get(name, []))
        for info in state.get("tools", {}).get(name, []):
            if info["name"] in enabled:
                _register_tool(name, label, info, tier_of(state, name, info["name"]))


hub = Hub()


def _transport(server: Server):
    """Build the transport for one server: a process on stdin and stdout, or Streamable HTTP with headers."""
    from contextlib import nullcontext

    if server.command:
        from mcp import StdioServerParameters

        params = StdioServerParameters(
            command=server.command, args=server.args, env=resolve(server.env), cwd=server.cwd
        )
        return nullcontext(params)
    return _http_transport(server)


class _http_transport:  # noqa: N801 - used like a function, as `async with`
    """An HTTP client with the resolved headers, wrapped in the SDK's Streamable HTTP transport."""

    def __init__(self, server: Server):
        """Resolve the headers now, so a missing variable fails before anything connects."""
        self.server, self.headers = server, resolve(server.headers)

    async def __aenter__(self):
        """Open the HTTP client and return the transport for `mcp.Client`."""
        import httpx2
        from mcp.client.streamable_http import streamable_http_client

        self.http = httpx2.AsyncClient(headers=self.headers, timeout=httpx2.Timeout(30.0, read=300.0))
        return streamable_http_client(str(self.server.url), http_client=self.http)

    async def __aexit__(self, *exc) -> None:
        """Close the HTTP client."""
        await self.http.aclose()


async def _answer(client, queue: asyncio.Queue) -> None:
    """Run each queued call on the session until a None arrives."""
    while (item := await queue.get()) is not None:
        tool, args, reply = item
        try:
            reply.set_result(await client.call_tool(tool, args))
        except Exception as e:  # noqa: BLE001 - the error goes back to the caller's thread
            reply.set_exception(e)


def _tool_info(t) -> dict:
    """Keep what the dashboard shows and the registry needs from one MCP Tool."""
    notes = t.annotations.model_dump(exclude_none=True) if t.annotations else {}
    return {"name": t.name, "description": t.description or "", "input_schema": t.input_schema, "annotations": notes}


def _register_tool(server: str, label: str, info: dict, tier: str) -> None:
    """Add one MCP tool to the shared registry under `<server>__<tool>`."""
    name, schema = info["name"], dict(info["input_schema"] or {"type": "object"})
    schema.setdefault("type", "object")
    tools.REGISTRY[f"{server}{SEPARATOR}{name}"] = tools.Tool(
        name=f"{server}{SEPARATOR}{name}",
        description=f"{info['description']} (MCP server '{server}', {label})",
        parameters=schema,
        tier=tier,
        fn=lambda **args: hub.call(server, name, args),
        examples=[],
    )


def _unregister(server: str) -> None:
    """Remove every tool of one server from the registry."""
    for key in [k for k in tools.REGISTRY if k.startswith(f"{server}{SEPARATOR}")]:
        del tools.REGISTRY[key]


def start() -> list[str]:
    """Connect every saved server that has at least one enabled tool. A server that fails is reported, not raised.

    Returns:
        list[str]: The servers that connected.
    """
    state = load()
    for name in [n for n, chosen in state.get("enabled", {}).items() if chosen and n in state["mcpServers"]]:
        try:
            hub.connect(name)
        except Exception as e:  # noqa: BLE001 - report and go on with the next server
            events.emit("mcp.error", server=name, error=f"{type(e).__name__}: {e}")
    return hub.connected()
