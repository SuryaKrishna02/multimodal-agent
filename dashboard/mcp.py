"""The MCP endpoints: paste a standard `mcpServers` config, see each server's label, pick tools and their tiers.

    GET  /api/mcp            the servers, their labels, whether each is connected, and each tool's state
    POST /api/mcp/config     replace the server list with a pasted config ({"text": "..."})
    POST /api/mcp/connect    start one server and list its tools ({"server": "reminders"})
    POST /api/mcp/enable     choose the tools the model may see ({"server": ..., "tools": [...]})
    POST /api/mcp/tier       set one tool's tier ({"server": ..., "tool": ..., "tier": "safe|confirm|blocked"})

A refused config (a literal secret, a bad entry) comes back as 400 with every problem listed.
"""

import json

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from agent import mcp_client

router = APIRouter(prefix="/api/mcp")


def _server_view(name: str, server: mcp_client.Server, state: dict) -> dict:
    """Describe one server and its tools for the page."""
    enabled = set(state.get("enabled", {}).get(name, []))
    return {
        "name": name,
        "label": server.label,
        "target": server.url or " ".join([str(server.command), *server.args]),
        "secrets": sorted({*server.env.values(), *server.headers.values()}),
        "connected": name in mcp_client.hub.connected(),
        "tools": [
            {**t, "enabled": t["name"] in enabled, "tier": mcp_client.tier_of(state, name, t["name"])}
            for t in state.get("tools", {}).get(name, [])
        ],
    }


@router.get("")
def overview() -> dict:
    """List the configured servers with their labels and tools.

    Returns:
        dict: `servers` and `config` (the pasted JSON, as saved: references, no secrets).
    """
    state = mcp_client.load()
    servers = mcp_client.servers()
    return {
        "servers": [_server_view(n, s, state) for n, s in servers.items()],
        "config": json.dumps({"mcpServers": state["mcpServers"]}, indent=1),
    }


@router.post("/config")
def set_config(body: dict) -> dict:
    """Replace the servers with a pasted config.

    Args:
        body (dict): `text`, the pasted JSON.

    Returns:
        dict: The new overview.

    Raises:
        HTTPException: 400 with every problem of the config.
    """
    try:
        mcp_client.set_config(str(body.get("text", "")))
    except mcp_client.ConfigError as e:
        raise HTTPException(400, str(e)) from e
    return overview()


@router.post("/connect")
async def connect(body: dict) -> dict:
    """Start one server and list its tools (npx or uvx may first download the server).

    Args:
        body (dict): `server`.

    Returns:
        dict: The new overview.

    Raises:
        HTTPException: 400 for a missing secret, 502 when the server does not start or answer.
    """
    try:
        await run_in_threadpool(mcp_client.hub.connect, str(body["server"]))
    except mcp_client.ConfigError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # noqa: BLE001 - show why the server did not start
        raise HTTPException(502, f"{body['server']} did not start: {type(e).__name__}: {e}") from e
    return overview()


@router.post("/enable")
def enable(body: dict) -> dict:
    """Choose which tools of a server the model may see.

    Args:
        body (dict): `server` and `tools` (names).

    Returns:
        dict: The new overview.
    """
    mcp_client.enable(str(body["server"]), [str(t) for t in body.get("tools", [])])
    return overview()


@router.post("/tier")
def tier(body: dict) -> dict:
    """Set the tier of one MCP tool.

    Args:
        body (dict): `server`, `tool` and `tier`.

    Returns:
        dict: The new overview.

    Raises:
        HTTPException: 400 for an unknown tier.
    """
    try:
        mcp_client.set_tier(str(body["server"]), str(body["tool"]), str(body["tier"]))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return overview()
